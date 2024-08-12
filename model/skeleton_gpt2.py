import torch
from torch import nn
from torch.nn import functional as F
from model.block.block import Block
from model.config import GPTConfig


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ModuleDict [E8]
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embeddings.
                wpe = nn.Embedding(config.block_size, config.n_embd), # Positional encodings.
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Number of blocks stacked on each other (E7)
                ln_f = nn.LayerNorm(config.n_embd), # Normalization layer.
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Final (linear) classifier head.

        # weight sharing scheme
        # TODO understand this part
        self.transformer.wte.weight= self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes the weights of the given module.

        Args:
            module (nn.Module): The module whose weights need to be initialized.

        Returns:
            None
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # We are using 2 * as we have the block and the MLP
                # TODO - understand this part
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size() # idx is of shape (B, T) = (batch size, sequence length) [F1]
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted." # block_size = maximum length of input sequences (block size is the maximum length of input sequences)

        """
        Positional embeddings [E9]
        """
        pos = torch.arange(start=0, end=T, step=1, dtype=torch.long, device=idx.device) # Shape (T)
        pos_emd = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)

        """
        Token embeddings [E10]
        """
        tok_emd = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emd + pos_emd # sum the token- and position embeddings

        # Iterate through each block
        for block in self.transformer.h:
            x = block(x)

        # Final layer normalization
        x = self.transformer.ln_f(x)

        # Final classifier head
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # --------------------------------------------------------------------------------
    # Extract the GPT model weights from huggingface
    """
    1) We are extracting the GPT model weights from huggingface.
    2) We reschape the weights to match the shape our our model class.
    """
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args) # Update our GPTConfig class
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        print("------\n\nThis is the converted hugging face model:\n\n------", model)
        return model