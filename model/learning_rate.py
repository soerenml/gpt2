import math

def get_lr(
        it: int,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float) -> float:
    # 1) linear warmup for warum_iter steps
    if it < warmup_steps:
        # we are using (it+1) as we start from 0
        return max_lr * (it+1) / warmup_steps
    # 2) if lr > lr_decay_iters, return min learning rate
    if it >= max_steps:
        return min_lr
    # 3) in between use cosine decay until min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff stars with 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)