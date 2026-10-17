from model import Transformer
from config import *
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import random
from dict import examples,source_words,target_words
import matplotlib.pyplot as plt


model = Transformer(source_words,target_words).to(DEVICE)
#定义损失函数和优化器
def build_scheduler(optimizer):
    """Warm-up（线性升温） + 余弦退火 组合学习率。"""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=MIN_LR/MAX_LR,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_EPOCHS - WARMUP_EPOCHS,
        eta_min=MIN_LR,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )
    return scheduler

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,weight_decay=WEIGHT_DECAY)
scheduler = build_scheduler(optimizer)
losses = []
model.train()

for epoch in range(TOTAL_EPOCHS):
    random.shuffle(examples)
    lr_now = scheduler.get_last_lr()[0]
    total_loss = 0.0
    for source, target in examples:
        logits = model(source,target[:-1])
        loss = criterion(logits,target[1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss / len(examples))
    if epoch % 10 == 0:
        print(f'epoch {epoch}, total loss: {total_loss / len(examples)}')
    scheduler.step()


plt.plot(range(1,101),losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


