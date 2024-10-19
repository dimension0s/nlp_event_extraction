# 4.主训练循环

import random
import numpy as np
import torch.nn as nn
import torch
import os
from transformers import AdamW,get_scheduler
from model import model
from collate_fn import train_dataloader,valid_dataloader
from train import train_loop
from test import test_loop

learning_rate = 1e-5
epoch_num = 3

def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)


loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(),lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader))

total_loss=0.
best_f1 = 0.
for epoch in range(epoch_num):
    print(f'epoch {epoch+1}/{epoch_num}\n---------------------------------')
    loss = train_loop(train_dataloader,model,loss_fn,optimizer,lr_scheduler,epoch+1)
    accuracy,precision,recall,f1 = test_loop(valid_dataloader,model,'Valid')
    if f1>best_f1:
        best_f1 = f1
        print('saving new weights...\n')
        torch.save(
            model.state_dict(),
            f'epoch_{epoch+1}_valid_accuracy_{accuracy:.4f}_valid_f1_{f1:.4f}_weights.bin')