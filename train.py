# 3.训练和验证/测试函数
# 3.1）训练函数

from tqdm.auto import tqdm
from device import device

def train_loop(dataloader,model,loss_fn,optimizer,lr_scheduler,epoch):
    total_loss = 0.
    total = 0
    model.train()

    progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0,2,1),y.long()) #  重点：.long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss/(step+1)
        progress_bar.set_description(f'epoch:{epoch},loss:{avg_loss:.4f}')
    return avg_loss