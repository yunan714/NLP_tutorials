from torch import nn
import torch

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        en = batch.src
        ge = batch.trg
        optimizer.zero_grad()
        output = model(en, ge)
        output = output[1:].view(-1, output.shape[-1])   #
        ge = ge[1:].view(-1)                             #去掉第一个<sos>无用字符
        loss = criterion(output, ge)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model ,iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():  #直接取消了梯度计算，节省大量时间
        for batch in iterator:
            en = batch.src
            ge = batch.trg

            output = model(en,ge, 0)
            output = output[1:].view(-1, output.shape[-1])
            ge = ge[1:].view(-1)

            loss = criterion(output,ge)
            epoch_loss+=loss.item()
    return epoch_loss/len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


