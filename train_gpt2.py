from dataclasses import dataclass
from email.headerregistry import DateHeader
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
from helper_functions import device_info

device = device_info()

# --------------------------------------------------------------------------------
# Attention mechanism module
from model.block.modules.attention import CasualSelfAttention

# --------------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) module
from model.block.modules.mlp import MLP

# --------------------------------------------------------------------------------
# GPT2 block
from model.block.block import Block

# --------------------------------------------------------------------------------
# Skeleton of the GPT model
from model.skeleton_gpt2 import GPT


# ---------------------------------------- Load model with model weights ----------------------------------------
num_return_sequences = 5 # number of sequences to generate.
max_length = 30 # maximum length of the generated sequences

#model = GPT.from_pretrained('gpt2') # load the model from the transformers library.
#model = GPT(GPTConfig()) # initialize the model with our GPTConfig class.
#model.eval() # we are in evaluation mode: we are not training the model, only using it to generate text.
#model.to(device) # we are moving all the model to the device at hand.


# --------------------------------------------------------------------------------
# Data pipeline
import tiktoken
from model.config import GPTConfig

class DataloaderLite():
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        B, T = 4, 32
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        #buf = buf.to(device) # to(device) moves the tensor to the device at hand. But it's not stateful (todo - add explaination)
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T) # y is basically x shifted by one token to the right
        self.current_position += B*T

        # In case we run out of data, we reset the position to the beginning of the data.
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# get logics

train_loader = DataloaderLite(B=4, T=32)
model = GPT(GPTConfig()) # initialize the model with our GPTConfig class.
model.to(device) # we are moving all the model to the device at hand.

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

for i in range(50):
    optimizer.zero_grad() # always set gradients to zero
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device) # to(device) moves the tensor to the device at hand. We are doing this here as we don't want to load the full dataset into the GPU memory.
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")


# --------------------------------------------------------------------------------
# Tokenization - first example

import tiktoken
enc = tiktoken.get_encoding('gpt2')
#tokens = enc.encode("Hello, I'm a language model,")
tokens = enc.encode("What is the meaning of life?")
tokens = torch.tensor(tokens, dtype=torch.long) # In this case it's (8,) tokens
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # Repeats this tensor along the specified dimensions.
x = tokens.to(device) # x is the idx for the forward function. I.e. the token we are feeding into the model.


# --------------------------------------------------------------------------------
# Generation
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print(f"\n\n Tokens to be feed into the model: \n\n {x}  \n\n with shape: {x.shape} \n\n")

while x.size(1) < max_length:
    with torch.no_grad(): # we are not training the model (no gradient calculation), only using it to generate text
        # We pass the current sequence x (tensor of token IDs) to the model
        # The model outputs the logits, which represent the unnormalized probabilities of the next token
        logits, loss = model(x)
        # We start out with a shape of 8 as our input tensor x contains 8 token IDs
        # After each iteration, we add a new token to the sequence, so the shape of x grows by 1
        print(logits.shape)
        # We extract the logits for the last token in the sequence (-1)
        # This is because we only care about predicting the next token based on the current context
        logits = logits[:, -1, :]
        # We apply a softmax function to the logits of the last token
        probs = F.softmax(logits, dim=-1)
        # torch.topk selects the top 50 probabilities (topk_probs) and their corresponding indices (topk_indices)
        # along the last dimension (vocabulary dimension).
        # this is an ordered tensor with the highest probabilities at the beginning
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # E[11]
        ix = torch.multinomial(topk_probs, 1)
        # We use torch.gather to extract the actual token ID from the top 50 indices (topk_indices) based on the sampled index (ix).
        # xcol becomes a one-dimensional tensor containing the sampled token ID.
        xcol = torch.gather(topk_indices, -1, ix)
        # We concatenate the sampled token ID to the current sequence x.
        x = torch.cat((x, xcol), 1)

# Print predicted tokens
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    # Here we decode the token IDs back to text.
    decoded = enc.decode(tokens)
    print(">", decoded)
