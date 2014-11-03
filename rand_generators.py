from random import randint, random
import numpy as np
import pylab as plt


def rand_vol(num_points, start_value, volatility):

    def func(old_price, volatility):
        rnd = random()
        pct_change = 2 * volatility * rnd
        if (pct_change > volatility):
            pct_change -= (2 * volatility)
        change_amount = old_price * pct_change
        return old_price + change_amount

    rdata = []
    price = start_value
    rdata.append(price)
    for i in range(num_points-1):
        price = func(price, volatility)
        rdata.append(price)

    return np.asarray(rdata)


def rand_range(nmbr_of_points, min_int, max_int):
    m = 10000.0
    rdata = []
    rnd = 0

    for i in range(nmbr_of_points):
        rnd = m + randint(min_int, max_int)
        rdata.append(rnd)
        m = rnd

    return np.asarray(rdata)

import numpy as np
from scipy.optimize import fsolve
from random import randint


def rand_fractal(dt1input, holder, lim):
    """Generates random multifractal data series with global dependence and long tails"""
    """Based on cartoons outlined at http://classes.yale.edu/fractals/RandFrac/Cartoon/Cartoon.html"""

    def recurse(x0, x1, y0, y1, direction):

        if (x1-x0) > lim:

            if direction == 1:
                a = x0+a0*(x1-x0)
                b = y0+b0*(y1-y0)
                c = x0+c0*(x1-x0)
                d = y0+d0*(y1-y0)
            if direction == 2:
                a = x0+(c0-a0)*(x1-x0)
                b = y0-(b0-d0)*(y1-y0)
                c = a+a0*(x1-x0)
                d = b+b0*(y1-y0)
            if direction == 3:
                a = x0+a0*(x1-x0)
                b = y0+b0*(y1-y0)
                c = a+a0*(x1-x0)
                d = b+b0*(y1-y0)

            output.append([a, b])
            output.append([c, d])

            recurse(x0, a, y0, b, randint(1, 3))
            recurse(a, c, b, d, randint(1, 3))
            recurse(c, x1, d, y1, randint(1, 3))

    x0 = 0.
    y0 = 0.
    x1 = 1.
    y1 = 1.

    def dt2constraint(x):
        return (dt1input**holder - x**holder + (1-dt1input-x)**holder - 1)

    dt1 = dt1input
    dt2 = fsolve(dt2constraint, 0.2)
    dt3 = 1-dt1-dt2
    dy1 = dt1**holder
    dy2 = -dt2**holder
    dy3 = dt3**holder

    a0 = dt1
    b0 = dy1
    c0 = dt1+dt2
    d0 = dy1+dy2

    output = []

    output.append([x0, y0])
    output.append([x1, y1])

    recurse(x0, x1, y0, y1, randint(1, 3))

    output = np.array(output)
    output = output[output[:, 0].argsort()]

    return np.array(output)
