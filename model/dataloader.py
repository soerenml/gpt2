import tiktoken
import torch

class DataloaderLite():
    def __init__(self, B, T, file_path="input.txt", print_data: bool = False):
        """
        Initializes the data loader by setting up parameters and loading data.
        """
        self.B = B # B is the batch size
        self.T = T # T is the length of the sequence
        self.current_position = 0
        self.print_data = print_data

        # Call the private method to load and process the data
        self._load_and_tokenize(file_path)


    def _load_and_tokenize(self, file_path):
        """
        Loads the text file, encodes it, and prints statistics.
        This is a private helper method.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        if self.print_data:
            print(f"Loaded text data from '{file_path}':")
            print(text[:1000])  # Print the first 1000 characters

        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))

        self.num_tokens = len(self.tokens)
        self.num_batches = self.num_tokens // (self.B * self.T)

        print(f"Loaded {self.num_tokens} tokens from '{file_path}'")
        print(f"Dataset has {self.num_batches} batches per epoch.")


    def next_batch(self):
        # TODO - understand this part
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        #buf = buf.to(device) # to(device) moves the tensor to the device at hand. But it's not stateful (todo - add explaination)
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T) # y is basically x shifted by one token to the right
        self.current_position += B*T

        # In case we run out of data, we reset the position to the beginning of the data.
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y