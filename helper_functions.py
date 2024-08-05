import torch

def device_info():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr (torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device