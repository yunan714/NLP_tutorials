## Encoder部分
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder,self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embeddings = nn.Embedding(input_dim,emb_dim)   ##??????
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout

    def forward(self, src):
        # src = [src sent len, batch_size]
        embedded = self.dropout(self.embeddings(src))
        # embedded [src sent len, batch_size, emb_dim]
        outputs , (hidden,cell) = self.rnn(embedded)
        return hidden, cell ## 不要outputs?

