import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from Configs import configs
from Data import get_data

config = configs()

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)   # 确定随机种子参数以便获得重现
torch.backends.cudnn.deterministic = True  # 当使用GPU为后台时，需要添加此操作

## 获取数据,以迭代器形式
train_iterator, valid_iterator, test_iterator, Ge_vocab, En_vocab = get_data()



