import tiktoken
import torch

def tokenizer(text: str, device: torch.device, nrs: int) -> torch.Tensor:
    """
    Tokenizes the input text and prepares it for model input.

    Args:
        text (str): The input text to be tokenized.
        device (str): The device to be used for tensor operations.
        nrs (int): The number of times to repeat the tokens tensor.

    Returns:
        torch.Tensor: The tokenized input text tensor.

    """
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long) # In this case it's (8,) tokens
    tokens = tokens.unsqueeze(0).repeat(nrs, 1) # Repeats this tensor along the specified dimensions.
    x = tokens.to(device) # x is the idx for the forward function. I.e. the token we are feeding into the model.
    return x