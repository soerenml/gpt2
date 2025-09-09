import torch

def device_info():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr (torch.backends, "mps") and torch.backends.mps.is_available(): # in case you are running on a mac
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n\n--- Using device: {device} ---\n\n")
    return device
