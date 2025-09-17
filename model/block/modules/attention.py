import torch
from torch import nn
from torch.nn import functional as F
import math

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # [D1]

        self.n_head = config.n_head # numer attention heads
        self.n_embd = config.n_embd # embedding dimensionality
        self.attention_type = config.attention_type # attention type

        """
        nn.Linear() [E1]
        By using a single linear layer to compute the concatenated Q, K, and V matrices in one go,
        we reduce computational overhead compared to applying three separate linear transformations.
        """
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1 # todo - what is this doing?

        """
        Mask [E2]
        """
        self.register_buffer( # buffer = tensor that is not updated during backpropagation
            name='bias',
            tensor=torch.tril( # creates a lower triangular matrix
                torch.ones(config.block_size, config.block_size)) # creates a matrix filled with ones (this is needed as torch.tril() needs a tensor as input)
                .view(1, 1, config.block_size, config.block_size) # reshape tensor to (1, 1, block_size, block_size) - block_size = maximum length of input sequences
        )
        """Shape of the Mask:
            •	(1, 1, config.block_size, config.block_size):
            •	The first dimension (batch size) is 1 because this mask is used for all sequences in the batch.
            •	The second dimension (number of heads) is 1 because the same mask is typically applied to all heads in the attention mechanism.
            •	The third and fourth dimensions are config.block_size to create a square matrix that masks out future positions in the sequence."""

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality. Here, [5, 17, 768]
        qkv = self.c_attn(x)

        """
        Splitting [E3]
        """
        q, k, v = qkv.split(self.n_embd, dim=2) # split the concatenated q, k, v matrices along dim = 2 (C = embedding dimensionality)

        # We devide the dimensionality of the embeddings by the number of heads as they will be concatenated later on. See [D1]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # -> (B, n_head, T, C)

        """
        Attention computation (E4)
        """
        # Flash attention
        if self.attention_type == 'flash':
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # E[14]

        else:
        # Regular attention
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