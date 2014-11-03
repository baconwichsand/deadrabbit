from random import randint, random
from math import sqrt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import pylab as plt
from Visualization import plot_IFS, plot_difference_IFS
from analyze import coarse_grain, driven_IFS
from DataManipulation import interpolate
from math import log


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
    for i in range(num_points):
        price = func(price, volatility)
        rdata.append(price)

    return rdata


def rand_range(nmbr_of_points, min_int, max_int):

    m = 10000
    rdata = []
    rnd = 0

    for i in range(nmbr_of_points):
        rnd = m + randint(min_int, max_int)
        rdata.append(rnd)
        m = rnd

    return rdata


def brownian(x0, n, dt, delta, out=None):
    """\
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def generate(dt1input, holder, lim):
    """Generates random multifractal data series with global dependence and long tails"""
    """Based on multifractal cartoons outlined at http://classes.yale.edu/fractals/RandFrac/Cartoon/Cartoon.html"""

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
    return output[output[:, 0].argsort()]
