import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SelfAttNet(nn.Module):
    def __init__(self, config):
        super(SelfAttNet, self).__init__()
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.bilstm = nn.LSTM(config.embedding_dim, config.hidden_dim,
                              config.num_layers, bidirectional=True)
        self.w1 = nn.Linear(config.hidden_dim * 2, config.da, bias=False)
        self.w2 = nn.Linear(config.da, config.r, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        N, L = x.size()
        emb_x = self.embedding(x) # (N, L, emb_dim)
        emb_x = emb_x.view(L, N, -1) # (L, N, emb_dim)
        h = Variable(torch.zeros(self.config.num_layers * 2, N, self.config.hidden_dim))
        c = h.clone()
        h0 = (h, c)
        H, _ = self.bilstm(emb_x, h0) # (L, N, hidden_dim*2)
        H = H.view(N, L, -1) # (N, L, hidden_dim*2)
        A = F.softmax(self.w2(self.tanh(self.w1(H))), dim=2) # (N, L, r)
        A = A.view(N, -1, L) # (N, r, L)
        M = torch.bmm(A, H) # (N, r, hidden_dim*2)
        return M, A
