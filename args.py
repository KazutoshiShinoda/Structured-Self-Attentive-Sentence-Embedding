import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Configure')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--num_embeddings', default=1000, type=int)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    