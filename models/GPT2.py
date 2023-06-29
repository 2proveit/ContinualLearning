import transformers, torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

class GPT2(nn.Module):
    def __init__(self,class_nums,model_name = 'gpt2'):
        super(GPT2).__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.return_dict  =False
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gpt2_model = AutoModel.from_pretrained(model_name, config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,class_nums)

    def forward(x):
        tokens = self.gpt2_tokenizer()
        GPT2_out = self.gpt2_model()