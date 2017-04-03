from functools import partial
import numpy as np


def triangle(double center, double width, double x):
    cdef double r = width / 2
    cdef double k = 1 / r

    cdef double left = center - r
    cdef double right = center + r

    if left <= x <= center:
        return k * (x - left) + 0
    elif center <= x <= right:
        return -k * (x - center) + 1
    else:
        return 0


def left_semi_triangle(double center, double width, double x):
    cdef double r = width / 2
    cdef double k = 1 / r

    cdef double left = center - r

    if left <= x <= center:
        return k * (x - left) + 0
    else:
        return 0


def right_semi_triangle(double center, double width, double x):
    cdef double r = width / 2
    cdef double k = 1 / r

    cdef double right = center + r

    if center <= x <= right:
        return -k * (x - center) + 1
    else:
        return 0


def composite_triangle(double center, double left_w, double right_w, double x):
    left_triangle = partial(left_semi_triangle, center, 2 * left_w)
    right_triangle = partial(right_semi_triangle, center, 2 * right_w)
    if center - left_w <= x <= center:
        return left_triangle(x)
    elif center <= x <= center + right_w:
        return right_triangle(x)
    else:
        return 0


def composite_triangular(double center, double left_w, double right_w):
    return partial(composite_triangle,
                   center,
                   left_w,
                   right_w)


def triangular(double center, double width):
    return partial(triangle, center, width)


def entropy(y):
    if len(set(y)) == 1:
        return 0

    positive_cnt = len(np.where(y == 1))
    negative_cnt = len(np.where(y == 2))

    p_p = positive_cnt / y.shape[0]
    n_n = negative_cnt / y.shape[0]

    return -(p_p * (1 - p_p) + n_n * (1 - n_n))


def entropy_by_nums(classes, total):
    ent = 0
    for c in classes:
        p = c / total
        if p != 0:
            ent -= p * (p - 1)

    return ent
