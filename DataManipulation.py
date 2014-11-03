import fractions
import numpy as np


# Removes x axis from data
def remove_x(data):
    if data.ndim == 1:
        return np.array(data)
    x = np.hsplit(data, 2)
    output = np.ravel(x[1])
    return np.array(output)


# Adds incremental x axis beginning at 'start'
def add_x(data, start=0):
    if data.ndim == 2:
        return np.array(data)
    x = np.arange(start, data.shape[0] + start)
    output = np.column_stack((x, data))
    return np.array(output)


# Normalize y data
def normalize(data):
    output = []
    mn = float(min(data))
    mx = float(max(data))
    my = mx - mn
    for i, x in enumerate(data):
        npt = ((x - mn) / my)
        output.append(npt)
    return np.array(output)

# performs linear interpolation, returns 1-d filled in array
def interpolate(data):
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
    return np.array(output)


#
# Takes two DSet and sets them to the same x-scale
#
def eq_scale(dset1, dset2):
    temp1 = add_x(dset1, start=1)
    temp2 = add_x(dset2, start=1)
    a = temp1.shape[0]
    b = temp2.shape[0]
    lcm = a * b / fractions.gcd(a, b)
    x = lcm / a
    y = lcm / b
    for i, n in enumerate(temp1):
        if i is not 0:
            temp1[i][0] *= x
    for j, m in enumerate(temp2):
        if j is not 0:
            temp2[j][0] *= y
    output1 = interpolate(temp1)
    output2 = interpolate(temp2)
    return output1, output2
