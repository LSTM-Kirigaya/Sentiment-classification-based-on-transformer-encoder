import torch
from torch.utils.data import Dataset
import json
import typing
from tqdm import tqdm
from colossalai.core import global_context
from colossalai.utils import get_dataloader
from collections import Counter
import numpy as np
import time
from config import BATCH_SIZE, DEBUG_BATCH_NUM

class SimpleMood4(Dataset):
    def __init__(self, label : torch.Tensor, token_id : torch.Tensor, lengths : torch.Tensor) -> None:
        super().__init__()
        assert token_id.shape[0] == label.shape[0]
        self.token_id  : torch.Tensor = token_id
        self.label : torch.Tensor = label
        self.lengths : torch.Tensor = lengths
    
    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.label[index],
            self.token_id[index],
            self.lengths[index]
        )
    
    def __len__(self):
        return self.label.shape[0]

def get_dataset(token_id_path : str, max_length : int = 512, test_size : float = 0.2 , preshrink : int = 120000, debug_mode : bool = False) -> typing.Tuple[Dataset, Dataset]:
    with open(token_id_path, "r", encoding="utf-8") as fp:
        data_json = json.load(fp)
    sample_num = len(data_json["label"]) 

    if debug_mode:
        # I want to simply debug to avoid disaster
        labels = []
        token_id = []
        all_labels = Counter(data_json["label"])
        # num for each class
        each_class_num = BATCH_SIZE * DEBUG_BATCH_NUM // len(all_labels)
        cur_index = 0
        for label in all_labels:
            labels += data_json["label"][cur_index : cur_index + each_class_num]
            token_id += data_json["token_id"][cur_index : cur_index + each_class_num]
            while cur_index < sample_num and data_json["label"][cur_index] != label:
                cur_index += 1
        data_json["label"] = labels
        data_json["token_id"] = token_id
        sample_num = len(labels)
    else:
        # my memory is not enough :(
        data_json["label"]    = data_json["label"][preshrink:]
        data_json["token_id"] = data_json["token_id"][preshrink:]
        sample_num -= preshrink

    token_id = []
    lengths = []

    iterrr = tqdm(range(sample_num))
    iterrr.set_description_str("getting dataset")
    for i in iterrr:
        id_seq = data_json["token_id"][i]
        if len(id_seq) < max_length:
            id_seq = id_seq[:-1] + [0] * (max_length - len(id_seq)) + [id_seq[-1]]
            length = len(id_seq)
        else:
            id_seq = id_seq[:max_length - 1] + [id_seq[-1]]
            length = max_length
        token_id.append(id_seq)
        lengths.append(length)

    label_tensor   = np.array(data_json["label"])
    token_id       = np.array(token_id)
    lengths        = np.array(lengths)
    
    indice = np.arange(sample_num)
    np.random.shuffle(indice)
    offline = int(sample_num * test_size)
    
    test_data_set = SimpleMood4(
        label=torch.LongTensor(label_tensor[indice[:offline]]),
        token_id=torch.LongTensor(token_id[indice[:offline]]),
        lengths=torch.LongTensor(lengths[indice[:offline]])
    )
    train_data_set = SimpleMood4(
        label=torch.LongTensor(label_tensor[indice[offline:]]),
        token_id=torch.LongTensor(token_id[indice[offline:]]),
        lengths=torch.LongTensor(lengths[indice[offline:]])
    )

    print("finish")

    return train_data_set, test_data_set

def length_to_attention_mask(lengths, max_length=512) -> torch.LongTensor:
    attention_mask = []
    for length in lengths:
        length = int(length)
        mask = [1] * length + [0] * (max_length - length)
        attention_mask.append(mask)
    return torch.LongTensor(attention_mask)

def now_str():
    return time.strftime(
        "%Y-%m-%d %H-%M-%S",
        time.localtime()
    )

if __name__ == "__main__":
    train_data_set, test_data_set = get_dataset(
        token_id_path="./data/token_id_4_mood.json",
        debug_mode=True
    )

    train_loader = get_dataloader(
        dataset=train_data_set,
        shuffle=True,
        batch_size=4,
        pin_memory=True,
        num_workers=4
    )

    test_loader = get_dataloader(
        dataset=test_data_set,
        shuffle=False,
        batch_size=4,
        pin_memory=True,
        num_workers=4
    )

    train_labels = []
    test_labels = []
    for labels, ids, lengths in tqdm(train_loader):
        mask = length_to_attention_mask(lengths)
        labels : torch.Tensor
        train_labels += labels.tolist()
        # print(ids.shape)
        # print(labels.shape)
        # print(mask.shape)

    for labels, ids, lengths in tqdm(test_loader):
        mask = length_to_attention_mask(lengths)
        labels : torch.Tensor
        test_labels += labels.tolist()
        # print(ids.shape)
        # print(labels.shape)
        # print(mask.shape)

    
    print("train allocate situation: {}".format(Counter(train_labels)))
    print("test  allocate situation: {}".format(Counter(test_labels )))