import math
import torch
from logs.logging_config import logger_debug

def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

def value_check(value, tag):
    if torch.isnan(value).any():
            logger_debug.debug(f'[NaN] {tag} {value}')
            print(f'[NaN] {tag} {value}')
    if torch.isinf(value).any():
            logger_debug.debug(f'[Inf] {tag} {value}')
            print(f'[Inf] {tag} {value}')