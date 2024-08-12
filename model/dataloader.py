import tiktoken
import torch

class DataloaderLite():
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        B, T = 4, 32
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        #buf = buf.to(device) # to(device) moves the tensor to the device at hand. But it's not stateful (todo - add explaination)
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T) # y is basically x shifted by one token to the right
        self.current_position += B*T

        # In case we run out of data, we reset the position to the beginning of the data.
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y