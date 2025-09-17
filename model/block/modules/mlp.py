from torch import nn

class MLP(nn.Module):
    """
    This module implements a simple feed-forward neural network block commonly used in transformer architectures.
    It consists of two linear layers with a GELU activation in between.

    Attributes:
        c_fc (nn.Linear): First linear layer projecting input from n_embd to 4 * n_embd dimensions.
        gelu (nn.GELU): GELU activation function with 'tanh' approximation.
        c_proj (nn.Linear): Second linear layer projecting back from 4 * n_embd to n_embd dimensions.

        config (object): Configuration object containing model parameters, specifically 'n_embd'.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (..., n_embd).

    Returns:
        torch.Tensor: Output tensor of shape (..., n_embd) after applying the MLP transformation.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # (1)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #todo - understand this part


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

"""
(1) - Computational Efficiency: The tanh approximation for GELU is faster to compute than the exact version,
which involves the cumulative distribution function (CDF) of the standard normal distribution.
For large-scale models and extensive training, these small computational savings can add up significantly.
"""