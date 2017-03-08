from functools import partial
from math import log



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
    
    classes = {}

    for d in y:
        if d in classes:
            classes[d] += 1
        else:
            classes[d] = 0
    
    ent = 0
    for c in classes:
        p = classes[c] / len(y)
        if p != 0:
            ent = ent - p * log(p, 2)
    
    return ent