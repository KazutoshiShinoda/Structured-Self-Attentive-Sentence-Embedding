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
            emb_x, P = model(train_x)


def test(config):
    model = SelfAttNet(config)
    model.eval()
    dataset = DataSet(cofig)


if __name__ = '__main__':
    main()