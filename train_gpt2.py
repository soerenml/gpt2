from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ---------------------------------------- Attention mechanism ----------------------------------------
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

        assert config.n_embd % config.n_head == 0 # % = modulo operator (31 % 10 = 1). n_embd must be fully divisible by n_head.

        """
        nn.Linear() [E1]
        """
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # output projection
        self.n_head = config.n_head # numer attention heads
        self.n_embd = config.n_embd # embedding dimensionality

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

        # TODO - Add core idea.
        qkv = self.c_attn(x) # by using a single linear layer to compute the concatenated Q, K, and V matrices in one go, we reduce computational overhead compared to applying three separate linear transformations.

        """
        Splitting [E3]
        """
        q, k, v = qkv.split(self.n_embd, dim=2) # split the concatenated q, k, v matrices along the last dimension (C = embedding dimensionality)

        # We devide the dimensionality of the embeddings by the number of heads as they will be concatenated later on.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)

        """
        Attention computation (E4)
        """
        # (Q@K)/sqrt(embedding lenth)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k.size(-1) is the length of the embeddings.

        """
        Mask
        At this point we have a full attention matrix, backward and forward.
        Nevertheless, we need to mask out the positions so only backward looking is possible.
        """
        # TODO - understand this part
        # The name 'bias' us the lower triangular matrix we created in the __init__ function.
        # It's a buffer, so it's not updated during backpropagation.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # matrix multiplication attention * values


        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


# ---------------------------------------- MLP module ----------------------------------------
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
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.

        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# ---------------------------------------- Block of the GPT model ----------------------------------------
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
        self.ln_1 = nn.LayerNorm(config.n_embd) # This part is different from the original transformer model.
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # Multi-layer perceptron or feed-forward network.

    def forward(self, x):
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # We apply layer normalization -> send normalized values to the attention block.
        # We add the attention block output to the input tensor (x = x + self.attn(self.ln_1(x))). todo - what would happend if negelct x +?
        x = x + self.attn(self.ln_1(x)) # The attention block is the only time the embeddings 'speak' to each other.
        # We apply layer normalization -> send normalized values to the feed-forward network.
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------- Configuration class for GPT model ----------------------------------------
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


# ---------------------------------------- Skeleton of the GPT model ----------------------------------------
class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model.

    Args:
        config (object): Configuration object containing model hyperparameters.

    Attributes:
        config (object): Configuration object containing model hyperparameters.
        transformer (nn.ModuleDict): Module dictionary containing various components of the transformer.
        lm_head (nn.Linear): Linear layer for the final classification head.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ModuleDict allows us to store a collection of modules in a single object.
        # We can access the modules using keys.
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Positional encodings.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Core part of the transformer.
            # h stand for hidden and contains the blocks we are stacking upon each other.
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer normalization.
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Final (linear) classifier head.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # todo - understand this function.
    def forward(self, idx):
        # idx is of shape (B, T)
        # T tokens in each of the B sequences.
        B, T = idx.size()
        # block_size = maximum length of input sequences
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        # forward the token and position embeddings.
        # arrange returns a 1D tensor with values from the start (0 in this case) to the end (T), excluding T.
        # the function is similar to Python’s built-in range function but returns a tensor instead of a list.
        pos = torch.arange(start=0, end=T, step=1, dtype=torch.long, device=idx.device) # Shape (T)

        # positional encoding of the transformer block
        pos_emd = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)

        # token embeddings of the transformer block
        tok_emd = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)

        # sum the token and position embeddings.
        x = tok_emd + pos_emd # sum the token and position embeddings.

        # forward the blocks to the transformer
        # the transformer block consists of several layers
        # with the loop function we iterate through each layer
        for block in self.transformer.h:
            x = block(x)

        # forward to the final layer normalization
        x = self.transformer.ln_f(x)

        # forward to the final classifier head
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

    #
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
        config = GPTConfig(**config_args)
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

num_return_sequences = 5
max_length = 30
model = GPT.from_pretrained('gpt2')
model.eval() # we are in evaluation mode: we are not training the model, only using it to generate text.
model.to('cpu') # we are moving all the model to GPU

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # In this case it's (8,) tokens
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # In this case it's (5, 8) tokens - five rows of eight tokens
# x is the idx for the forward function
x = tokens.to('cpu')
print(x.device)

# todo - understand this part.
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), 1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
