# 1.3)构建批次函数

import numpy as np
import torch
from transformers import BertTokenizer,AutoTokenizer
from label_dict import label2id,id2label
from torch.utils.data import DataLoader
from data import train_data,valid_data

checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples):
    batch_text,batch_tags = [],[]
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_tags.append(sample['labels'])
    batch_inputs = tokenizer(
        batch_text,
        padding=True,
        truncation=True,
        return_tensors = "pt")
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for s_idx, text in enumerate(batch_text):
        encoding = tokenizer(text,truncation=True)
        batch_label[s_idx][0] = -100
        batch_label[s_idx][len(encoding.tokens())-1:] = -100
        for tag,_,char_start,char_end in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start:token_end] = label2id[tag]
    return batch_inputs,torch.tensor(batch_label)

train_dataloader = DataLoader(train_data,batch_size=4,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=4,shuffle=False,collate_fn=collote_fn)

# 打印数据集测试
batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)