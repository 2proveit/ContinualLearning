import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertClass(nn.Module):
    def __init__(self, args):
        super(BertClass, self).__init__()
        self.args = args
        config = BertConfig.from_pretrained(self.args.model_name)
        config.return_dict = False
        self.bert = BertModel.from_pretrained(self.args.model_name, config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
