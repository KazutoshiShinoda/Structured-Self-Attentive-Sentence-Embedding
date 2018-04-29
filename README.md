# Structured-Self-Attentive-Sentence-Embedding

Implementation of 'A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING' proposed by Lin et al. at ICLR 2017.

## How to use

```python:
from model import SelfAttNet
from args import get_args

config = get_args()
model = SelfAttNet(config)

emb_x, P = model(x)

## calculate loss for a specific task ##
loss = ...
alpha = 0.3

(loss + alpha * P).backword()

```

## main.py is not completed

At first, I try to implement main.py, but I came to think it was bothering me.

Please choose a specific task and implement it by yourself.
