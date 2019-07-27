import torch
from torch import nn
import random

class Decoder(nn.Module):
    def __init__(self,output_dim,emb_dim,hid_dim,n_layers,drop_out):
        super(Decoder,self).__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = drop_out

        self.embedding = nn.Embedding(output_dim, emb_dim) #输出语种的embedding

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=drop_out)

        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0) #在0维增加一个维度

        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder,self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embeddings = nn.Embedding(input_dim,emb_dim)   ##输入语种的embedding
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embeddings(src))   ## 为什么对于输入要进行dropout？
        outputs , (hidden,cell) = self.rnn(embedded)
        return hidden, cell ## 直接将hidden和cell输入后面的解码器中


class Seq2seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super(Seq2seq,self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim , \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0,:]
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        return outputs

