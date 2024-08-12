from torch import nn
from model.block.modules.attention import CasualSelfAttention
from model.block.modules.mlp import MLP


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
        self.mlp  = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x