from functools import partial
from math import log
import numpy as np



def triangle(center, width, name, x):
    r = width / 2
    k = 1 / r
    
    left = center - r
    right = center + r

    if x == 'name':
        return name
    else:
        x = float(x)

        if left <= x <= center:
            return k * (x - left) + 0
        elif center <= x <= right:
            return -k * (x - center) + 1
        else:
            return 0

def triangular(center, width, name='x'):
    return partial(triangle, center, width, name)

def entropy(y):
    if len(set(y)) == 1:
        return 0
    
    positive_cnt = len( np.where(y == 1) )
    negative_cnt = len( np.where(y == 2) )

    p_p = positive_cnt / y.shape[0]
    n_n = negative_cnt / y.shape[0]

    return -(p_p * (1 - p_p) + n_n * (1 - n_n))

def entropy_by_nums(classes, total):
    ent = 0
    for c in classes:
        p = c / total
        if p != 0:
            ent = ent - p * (p - 1)

    return ent
