import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from torchstat import stat
from Configs import configs
from Processed_data import get_data
from Model import *
from Utils import *
from Execute import *

# 基础配置类
config = configs()

# 确定随机种子参数以便获得重现
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # 当使用GPU为后台时，需要添加此操作

## 获取以迭代器形式获取数据
train_iterator, valid_iterator, test_iterator, Ge_vocab, En_vocab = get_data()

INPUT_DIM = len(En_vocab)
OUTPUT_DIM = len(Ge_vocab)

## 构建网络
enc = Encoder(INPUT_DIM,config.EN_EMB_DIM,config.HID_DIM,config.N_LAYERS,config.EN_DROPOUT)
dec = Decoder(OUTPUT_DIM,config.EN_EMB_DIM,config.HID_DIM,config.N_LAYERS,config.GE_DROPOUT)
model = Seq2seq(enc,dec,config.DEVICE).cuda()

## 参数初始化
model.apply(init_weights)

## 构建模型优化器和损失函数
optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
PAD_IDX = Ge_vocab.stoi['<pad>'] #获取padding对应的index
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) #保证target部分的pad部分不影响最终分数。

best_valid_loss = float("inf")

for epoch in range(config.N_EPOCHES):
    start_time = time.time()
    ## 同时进行训练和交叉验证
    train_loss = train(model, train_iterator, optimizer, criterion, config.CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time,end_time)

    if valid_loss < best_valid_loss:
        # 每一阶段军保存取得更好泛化误差的模型，为了简便只储存参数
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

## 读取参数，重构模型
model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
