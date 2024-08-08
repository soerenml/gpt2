from torch import nn

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