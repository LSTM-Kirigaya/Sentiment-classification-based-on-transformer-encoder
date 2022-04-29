import torch 
from torch import nn 
from transformers import BertModel, BertConfig, BertTokenizer

class BertMoodClassifier(nn.Module):
    pretrained_path = "/mnt/e/Pretrained/Bert/bert-base-chinese/"
    def __init__(self, num_label : int, bert_config : dict, pretrained_bert : bool = True) -> None:
        super().__init__()
        if pretrained_bert:
            self.bert_model = BertModel.from_pretrained(self.pretrained_path, config = bert_config)
        else:
            self.bert_model = BertModel.from_pretrained(config = bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size, num_label)
            
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_out = self.bert_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids            
        )
        
        out : torch.Tensor = self.classifier(bert_out[1])
        return out.softmax(dim=-1)

if __name__ == "__main__":
    model_name = 'bert-base-chinese'
    config = BertConfig.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config.output_hidden_states = True
    config.output_attentions = True
    
    model = BertMoodClassifier(
        num_label=4,
        bert_config=config,
        pretrained_bert=True
    )

    demo1 = "你好，今天天气不错"
    demo2 = "你好，今天天气不错"
    demo3 = "你好，今天天气不错"

    res1 = tokenizer.encode_plus(demo1)
    res2 = tokenizer.encode_plus(demo2)
    res3 = tokenizer.encode_plus(demo3)

    out = model.forward(
        input_ids=torch.tensor([res1["input_ids"], res2["input_ids"], res3["input_ids"]]),
        attention_mask=torch.tensor([res1["attention_mask"], res2["attention_mask"], res3["attention_mask"]]),
    )
    print(out)