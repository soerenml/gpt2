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
        assert config.n_embd % config.n_head == 0 # % is the modulo operator (31 % 10 = 1). n_embd must be fully divisible by n_head.
        # key, query, value projections for all heads but in a batch.
        # the nn.Linear layer computes a linear transformation of the input data.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Block size is the maximum length of input sequences.
        # Tril returns the lower triangular part of the matrix (2-D tensor)
        self.register_buffer(
            name='bias',
            tensor=torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size) # reshape the tensor to (1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality.
        # 'nh' is the number of heads, 'hs' ist the head size, 'C' (number of channels) = nh * hs.
        # By using a single linear layer to compute the concatenated Q, K, and V matrices in one go, we reduce computational overhead compared to applying three separate linear transformations.
        # This combined computation is more efficient and helps in leveraging hardware accelerations like GPUs and TPUs effectively.
        # todo - visualize this.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for al the queries and keys).
        # (Q@K)/sqrt(embedding lenth)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k.size(-1) is the length of the embeddings.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs
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

model = GPT.from_pretrained('gpt2')
print("did't crash yay")