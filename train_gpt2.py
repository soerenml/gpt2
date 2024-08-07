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
# Configuration class for GPT model
@dataclass
class GPTConfig:
    """
    Configuration class for GPT model.

    Attributes:
        block_size (int): The maximum length of input sequences.
        vocab_size (int): The size of the vocabulary.
        n_layer (int): The number of layers in the model.
        n_head (int): The number of attention heads in the model.
        n_embd (int): The dimensionality of the embeddings.
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# --------------------------------------------------------------------------------
# Attention mechanism module
class CasualSelfAttention(nn.Module):
    """
    CasualSelfAttention module performs self-attention operation on the input tensor.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        c.attn (nn.Linear): Linear layer for key, query, value projections for all heads.
        c_proj (nn.Linear): Linear layer for output projection.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimensionality.
        bias (torch.Tensor): Lower triangular bias matrix for attention mask.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # % = modulo operator (31 % 10 = 1). # See [D1]

        self.n_head = config.n_head # numer attention heads
        self.n_embd = config.n_embd # embedding dimensionality

        """
        nn.Linear() [E1]
        By using a single linear layer to compute the concatenated Q, K, and V matrices in one go,
        we reduce computational overhead compared to applying three separate linear transformations.
        """
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # output projection

        """
        Mask [E2]
        """
        self.register_buffer( # buffer = tensor that is not updated during backpropagation
            name='bias',
            tensor=torch.tril( # creates a lower triangular matrix
                torch.ones(config.block_size, config.block_size)) # creates a matrix filled with ones (this is needed as torch.tril() needs a tensor as input)
                .view(1, 1, config.block_size, config.block_size) # reshape tensor to (1, 1, block_size, block_size) - block_size = maximum length of input sequences
        )
        """
            Shape of the Mask:
            •	(1, 1, config.block_size, config.block_size):
            •	The first dimension (batch size) is 1 because this mask is used for all sequences in the batch.
            •	The second dimension (number of heads) is 1 because the same mask is typically applied to all heads in the attention mechanism.
            •	The third and fourth dimensions are config.block_size to create a square matrix that masks out future positions in the sequence.
        """

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality. Here, [5, 17, 768]
        qkv = self.c_attn(x)

        """
        Splitting [E3]
        """
        q, k, v = qkv.split(self.n_embd, dim=2) # split the concatenated q, k, v matrices along the last dimension (C = embedding dimensionality)

        # We devide the dimensionality of the embeddings by the number of heads as they will be concatenated later on. See [D1]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)

        """
        Attention computation (E4)
        """
        # (Q@K)/sqrt(embedding lenth)
        # This is the part which is most computationally expensive.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k.size(-1) is the length of the embeddings.

        """
        Mask (E5)
        At this point we have a full attention matrix, backward and forward.
        Nevertheless, we need to mask out the positions so only backward looking is possible.
        """
        # The name 'bias' us the lower triangular matrix we created in the __init__ function.
        # It's a buffer, so it's not updated during backpropagation.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # matrix multiplication attention * values - here we are going to use kv-caching in the future.

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y) # output projection
        return y


# --------------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) module
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        c_fc (nn.Linear): Linear layer for the fully connected operation.
        gelu (nn.GELU): GELU activation function.
        c_proj (nn.Linear): Linear layer for the projection operation.

    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# --------------------------------------------------------------------------------
# GPT2 block
class Block(nn.Module):
    """
    A block in the GPT-2 model.

    Args:
        config (GPT2Config): The configuration object for the GPT-2 model.

    Attributes:
        ln_1 (nn.LayerNorm): Layer normalization module for the first layer.
        attn (CausalSelfAttention): Causal self-attention module.
        ln_2 (nn.LayerNorm): Layer normalization module for the second layer.
        mlp (MLP): Multi-layer perceptron or feed-forward network.

    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # Different from the original transformer model: We do layer normalization before the attention block.
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# --------------------------------------------------------------------------------
# Skeleton of the GPT model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ModuleDict [E8]
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embeddings.
                wpe = nn.Embedding(config.block_size, config.n_embd), # Positional encodings.
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Number of blocks stacked on each other (E7)
                ln_f = nn.LayerNorm(config.n_embd), # Normalization layer.
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Final (linear) classifier head.

        # weight sharing scheme
        # TODO understand this part
        self.transformer.wte.weight= self.lm_head.weight


    def forward(self, idx, targets=None):
        B, T = idx.size() # idx is of shape (B, T) = (batch size, sequence length)

        """
        Block size vs. sequence length
        Sequence length (T) is the length of the input sequence, which can be less than or equal to the block size.
        The block size is the maximum length of input sequences that the model can process.
        """
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted." # block_size = maximum length of input sequences (block size is the maximum length of input sequences)

        """
        Positional embeddings [E9]
        """
        pos = torch.arange(start=0, end=T, step=1, dtype=torch.long, device=idx.device) # Shape (T)
        pos_emd = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)

        """
        Token embeddings [E10]
        """
        tok_emd = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emd + pos_emd # sum the token- and position embeddings

        # Feed embeddings through the transformer blocks
        # As we have n blocks, we use a loop function to iterate through each block
        for block in self.transformer.h:
            x = block(x)

        # forward to the final layer normalization
        x = self.transformer.ln_f(x)

        # forward to the final classifier head
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args) # Update our GPTConfig class
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


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
