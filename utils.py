import torch
import random
import numpy as np
import builtins
import fcntl

def print(*args, is_print_rank=True, **kwargs):
    """ solves multi-process interleaved print problem """
    if not is_print_rank: return
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def to_readable_format(num, precision=3):
    """Format a number with a T/B/M/K suffix (e.g. 4096000 -> 4.096M)."""
    num = float(num)
    for power, suffix in [(12, "T"), (9, "B"), (6, "M"), (3, "K")]:
        if abs(num) >= 10**power:
            return f"{num / 10**power:.{precision}f}{suffix}"
    if num.is_integer():
        return str(int(num))
    return f"{num:.{precision}f}"