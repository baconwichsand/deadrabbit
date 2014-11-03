"""
Version 0
Set of funcations that normalize and resize data sets and subsets
"""
import fractions
import numpy as np


def remove_x(data):
    """Removes x axis from data"""
    if data.ndim == 1:
        return np.array(data)
    x = np.hsplit(data, 2)
    output = np.ravel(x[1])
    return np.array(output)


def add_x(data, start=0):
    """Adds incremental x axis beginning at 'start'"""
    if data.ndim == 2:
        return np.array(data)
    x = np.arange(start, data.shape[0] + start)
    output = np.column_stack((x, data))
    return np.asarray(output)


def normalize(dset):
    """
    Normalize y data
    Input must be floats only.
    """
    mn = min(dset)
    mx = max(dset)
    my = mx - mn
    output = []
    for x in dset:
        npt = ((x - mn) / my)
        output.append(npt)
    return np.asarray(output)


def interpolate(data):
    """Performs linear interpolation, returns 1-d filled in array"""
    temp = []
    for i, x in enumerate(data):
        startp = data[i]
        if np.array_equal(startp, data[data.shape[0]-1]):
            temp.append(startp)
            break
        else:
            endp = data[i+1]
        temp.append(startp)
        j = int(startp[0])+1
        while j < int(endp[0]):
            x1 = float(startp[0])
            y1 = float(startp[1])
            x2 = float(endp[0])
            y2 = float(endp[1])
            a = float(j)
            point = [a, y1 + (y2 - y1) * ((a - x1) / (x2 - x1))]
            temp.append(point)
            j += 1
    output = []
    for x in temp:
        output.append(x[1])
    return np.asarray(output)

import pdb


def eq_xscale(trng_dset, trgt_dset):
    """Takes two DSet and sets them to the same x-scale"""
    pdb.set_trace()
    temp1 = add_x(trng_dset, start=1)
    temp2 = add_x(trgt_dset, start=1)
    a = temp1.shape[0]
    b = temp2.shape[0]
    lcm = a * b / fractions.gcd(a, b)
    x = lcm / a
    y = lcm / b
    temp1[:, 0] = temp1[:, 0].dot(x)
    temp2[:, 0] = temp2[:, 0].dot(y)
    output = interpolate(temp1), interpolate(temp2)
    return np.asarray(output)
