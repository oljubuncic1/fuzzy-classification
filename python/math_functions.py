from functools import partial



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