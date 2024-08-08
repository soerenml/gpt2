from dataclasses import dataclass

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