import builtins
import random

import numpy as np
import torch


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print(*args, is_print_rank=True, **kwargs):
    """ 
    
    if is_print_rank:
        builtins.print(*args, **kwargs)