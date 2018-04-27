import torch
import torch.nn as nn
from torch.autograd import Variable
from model import SelfAttNet
from args import get_args
from utils import DataSet


def main():
    config = get_args()
    if config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
        

def train(config):
    model = SelfAttNet(config)
    model.train()
    dataset = DataSet(cofig)
    
    for epoch in range(config.num_epochs):
        for data in dataset:
            train_x, train_y = data
            train_x = Variable(convert_words(train_x))
            true_y = Variable(train_y)
            emb_x, A = model(train_x)
            
            P = torch.sqrt(torch.sum(torch.bmm(A, A.transpose(1, 2)) - Variable(torch.eye(self.config.r))**2))
            
            
    

def test(config):
    model = SelfAttNet(config)
    model.eval()
    dataset = DataSet(cofig)

    for data in dataset:
        model()

if __name__ = '__main__':
    main()