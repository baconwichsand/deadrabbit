import math
import numpy as np


def align_manual(rdata, param):

    # import pdb
    # pdb.set_trace()

    ndata = []
    for x in rdata:
        ndata.append(x)

    x = len(rdata)-1
    y0 = ndata[0]
    y1 = ndata[x] - y0
    y2 = ndata[x] + param - y0
    alpha = math.atan(y1/x)
    gamma = math.atan(y2/x)
    for i, x in enumerate(ndata):
        if i == 0:
            pass
        else:
            y1 = i * math.tan(alpha)
            y2 = i * math.tan(gamma)
            ndata[i] += (y2 - y1)
    return ndata


#
#  Fits a time series to another based on endpoints
#
#  Inputs:
#           - list [] of y values, desired pattern
#           - list [] of y values, data to be fit to pattern
#
def align(pattern, rdata):

    ndata = []
    for x in rdata:
        ndata.append(x)

    diff = pattern[0] - rdata[0]
    if diff != 0:
        for i, x in enumerate(ndata):
            ndata[i] += diff
    x = len(pattern)-1
    y0 = ndata[0]
    y1 = ndata[x] - y0
    y2 = pattern[x] - y0
    alpha = math.atan(y1/x)
    gamma = math.atan(y2/x)
    for i, x in enumerate(ndata):
        if i == 0:
            pass
        else:
            y1 = i * math.tan(alpha)
            y2 = i * math.tan(gamma)
            ndata[i] += (y2 - y1)
    return ndata


def rotation(pattern, angle):
    patt = []
    for i, x in enumerate(pattern):
        patt.append([float(i)] + [pattern[i]])
        if i == 10:
            break
    ang = math.radians(angle)
    tmatrix = [[math.cos(ang), math.sin(ang)], [-1*math.sin(ang), math.cos(ang)]]
    product = np.dot(patt, tmatrix)
    return product


#
# Shear around mid x or y plane
# If axis = 'x', factor must be int!
#
def shear(pattern, factor):
    patt = []
    for i, x in enumerate(pattern):
        patt.append([float(i)] + [pattern[i]])
    # if axis == 'x':
    #     if not isinstance(factor, int):
    #         print 'Factor must be INT'
    #     line = max(pattern) / 2
    #     for j, x in enumerate(patt):
    #         patt[j][1] -= line
    #     tmatrix = [[1, 0], [factor, 1]]
    #     product = np.dot(patt, tmatrix)
    #     for k, x in enumerate(product):
    #         product[k][1] += line
    # if axis == 'y':
    line = len(pattern) / 2
    for j, x in enumerate(patt):
        patt[j][0] -= line
    tmatrix = [[1, factor], [0, 1]]
    product = np.dot(patt, tmatrix)
    for k, x in enumerate(product):
        product[k][0] += line
    output = []
    for l, x in enumerate(product):
        output.append(product[l][1])
    return output


def stretch(matrix, factor):
    tmatrix = [[0, 0], [0, factor]]
    return np.dot(matrix, tmatrix)
