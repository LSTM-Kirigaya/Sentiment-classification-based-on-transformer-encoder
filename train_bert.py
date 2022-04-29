import colossalai
from colossalai.core import global_context
from colossalai.utils import get_dataloader
from colossalai.logging import get_dist_logger
from transformers import BertConfig
import torch
from tqdm import tqdm
import os
from config import LEARNING_RATE

from bert import BertMoodClassifier
from dataset import get_dataset, length_to_attention_mask, now_str

token_id_path = "./data/token_id_4_mood.json"
model_name = 'bert-base-chinese'
checkpoint_dir = "model"

def main():
    colossalai.launch_from_torch(config="/home/kirigaya/Project/Sentiment-classification-based-on-transformer-encoder/config.py")
    
    if not os.path.exists(checkpoint_dir):
        os.mkdirs(checkpoint_dir)

    train_dataset, test_dataset = get_dataset(
        token_id_path=token_id_path, 
        max_length=512, 
        test_size=0.2,
        debug_mode=True
        # preshrink=361200
    )

    model = BertMoodClassifier(
        num_label=4,
        bert_config=BertConfig.from_pretrained(model_name),
        pretrained_bert=True
    )

    train_loader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        batch_size=global_context.config.BATCH_SIZE
    )

    test_loader = get_dataloader(
        dataset=test_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        batch_size=global_context.config.BATCH_SIZE
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    engine, train_data_loader, test_data_loader, _ = colossalai.initialize(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        test_dataloader=test_loader
    )

    logger = get_dist_logger(name="train_bert")

    # just for test
    checkpoint_path = os.path.join(checkpoint_dir, "test.pth")
    torch.save(obj={
        "state_dict" : engine.model.state_dict()
    }, f=checkpoint_path)


    for epoch in range(global_context.config.NUM_EPOCHS):
        engine.train()
        if global_context.get_global_rank() == 0:
            real_train_dl = tqdm(train_data_loader)
        else:
            real_train_dl = train_data_loader
        for labels, ids, lengths in real_train_dl:
            ids = ids.cuda()
            labels = labels.cuda()
            masks = length_to_attention_mask(lengths)
            masks = masks.cuda()
            out : torch.Tensor = engine(ids, masks)

            engine.zero_grad()
            train_loss : torch.Tensor = engine.criterion(out, labels)
            engine.backward(train_loss)
            engine.step()
   
        engine.eval()
        correct_num = 0
        total = 0
        for labels, ids, lengths in test_data_loader:
            ids = ids.cuda()
            labels = labels.cuda()
            masks = length_to_attention_mask(lengths)
            masks = masks.cuda()
            with torch.no_grad():
                out : torch.Tensor = engine(ids, masks)
                test_loss : torch.Tensor = engine.criterion(out, labels)
                pre_lab = out.argmax(dim=-1)
            correct_num += int(torch.sum(pre_lab == labels))
            total += labels.shape[0]
        # save checkpoint
        stamp = now_str() + ".pth"
        checkpoint_path = os.path.join(checkpoint_dir, stamp)

        torch.save(obj={
            "state_dict" : engine.model.state_dict()
        }, f=checkpoint_path)

        logger.info(
            message="Epoch : {} - train loss : {} - test loss : {} - acc : {}".format(
                epoch,
                round(train_loss.item(), 5),
                round(test_loss.item(), 5),
                round(correct_num / total, 5)
            ),
            ranks=[0]
        )

if __name__ == "__main__":
    main()
