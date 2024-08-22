import torch
from torch.nn import functional as F
import tiktoken
from helper_functions import device_info
import time
from dataclasses import fields

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = device_info()


# --------------------------------------------------------------------------------
# Load model configuration
from model.config import GPTConfig

num_return_sequences = 5 # number of sequences to generate.
max_length = 30 # maximum length of the generated sequences

print("--- Model configuration ---\n")
for field in fields(GPTConfig()):
    print(f"{field.name}: {field.default}")

# --------------------------------------------------------------------------------
# Skeleton of the GPT model
from model.skeleton_gpt2 import GPT


# --------------------------------------------------------------------------------
# Load model with hugging face weights
model_hf = GPT.from_pretrained(model_type='gpt2', print_model=False) # load the model from the transformers library.
model_hf = GPT(GPTConfig()) # initialize the model with our GPTConfig class.
model_hf.eval() # we are in evaluation mode: we are not training the model, only using it to generate text.
model_hf.to(device) # we are moving all the model to the device at hand.


# --------------------------------------------------------------------------------
# Train model from scratch
from model.dataloader import DataloaderLite

train_loader = DataloaderLite(B=32, T=1024)
# speed improvement with float32
torch.set_float32_matmul_precision('high') # change quantization [E14]

model = GPT(GPTConfig()) # initialize the model with our GPTConfig class.
model.to(device) # we are moving all the model to the device at hand.

if device.type != 'mps':
        model = torch.compile(model) # speed improvement

# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device) # to(device) moves the tensor to the device at hand. We are doing this here as we don't want to load the full dataset into the GPU memory.
    optimizer.zero_grad() # always set gradients to zero
    # TODO - run this on a A100
    # Check device. If not an A100 don't use autocast
    if device.type == 'mps':
        logits, loss = model(x, y)
    else:
        # speed improvement with autocast
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    print(f"step {i}, loss: {loss.item()}, dt:{dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")


# --------------------------------------------------------------------------------
# Tokenization - first example
from model.tokenizer import tokenizer

x = tokenizer(
    text="Hello, I'm a language model,",
    device=device,
    nrs=num_return_sequences)


# --------------------------------------------------------------------------------
# Generation
print(f"\n\n Tokens to be feed into the model: \n\n {x}  \n\n with shape: {x.shape} \n\n")

model_type = 'hff'
if model_type == 'hf':
    model = model_hf
    print("Hugging face model")
else:
    model = model
    print("Trained model")

while x.size(1) < max_length:
    with torch.no_grad(): # we are not training the model (no gradient calculation), only using it to generate text
        # We pass the current sequence x (tensor of token IDs) to the model
        # The model outputs the logits, which represent the unnormalized probabilities of the next token
        logits, loss = model(x)
        # We start out with a shape of 8 as our input tensor x contains 8 token IDs
        # After each iteration, we add a new token to the sequence, so the shape of x grows by 1
        print(logits.shape)
        # We extract the logits for the last token in the sequence (-1)
        # This is because we only care about predicting the next token based on the current context
        logits = logits[:, -1, :]
        # We apply a softmax function to the logits of the last token
        probs = F.softmax(logits, dim=-1)
        # torch.topk selects the top 50 probabilities (topk_probs) and their corresponding indices (topk_indices)
        # along the last dimension (vocabulary dimension).
        # this is an ordered tensor with the highest probabilities at the beginning
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # E[11]
        ix = torch.multinomial(topk_probs, 1)
        # We use torch.gather to extract the actual token ID from the top 50 indices (topk_indices) based on the sampled index (ix).
        # xcol becomes a one-dimensional tensor containing the sampled token ID.
        xcol = torch.gather(topk_indices, -1, ix)
        # We concatenate the sampled token ID to the current sequence x.
        x = torch.cat((x, xcol), 1)

# Print predicted tokens
enc = tiktoken.get_encoding('gpt2')
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    # Here we decode the token IDs back to text.
    decoded = enc.decode(tokens)
    print(">", decoded)