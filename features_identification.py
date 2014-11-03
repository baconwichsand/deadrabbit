import math
import numpy as np
import data_manipulation
from io_operations import Config


class FeatureIdentifier():

    #
    # Create FeatureIdentifier object
    #
    def __init__(self, data):
        self.data = data

    #
    # Returns 2-d array using a simplification algo
    # Algo options:
    #    -- zigzag (Default)
    #    -- fractal resampling
    #
    def simplify(self, param, algo='zigzag'):
        output = []
        temp = data_manipulation.normalize(self.data)
        if algo == 'zigzag':
            output = self.zigzag(temp, param)
        if algo == 'fractal':
            output = self.fractal(temp, param)
        return np.asarray(output)

    #
    #  FRACTAL RESAMPLING
    #
    #  Lossy compression algorithm for time series data
    #
    #  Main function, constructs output []
    #  and starts recursive process
    #
    #  Inputs:
    #          -  list of normalized floats
    #          -  max error value allowed by compression
    #
    #  Output:
    #          - SORTED list [] of [x, y] float pairs
    #
    def fractal(self, series, maxerror):
        rdata = []
        for i, x in enumerate(series):
            rdata.append([float(i+1)] + [series[i]])
        output = []

        #
        #  Recursive process:
        #
        #  1. Add endpoints to output
        #  2. Get midpoint
        #  3. Get point corresponding to x-value of
        #      midpoint on linear interpolation between endpoints
        #  4. Calc y-difference between original
        #      and interpolated points
        #  5. If difference > maxerror, add midpoint to output, if not,
        #      move midpoint 1 to the left and repeat steps 3.-5.
        #  6. Repeat steps 2.-5. for [startpoint, midpoint] and [midpoint, endpoint]
        #
        def process(series, maxerror, output):
            if not series:
                return
            endp = series[len(series)-1]
            startp = series[0]
            if not output:
                output.append(endp)
                output.append(startp)
            n = 0
            added = False
            while added == False:
                midp = series[(len(series) / 2) - n]
                x1 = float(startp[0])
                y1 = float(startp[1])
                x2 = float(endp[0])
                y2 = float(endp[1])
                a = float(midp[0])
                point = [a, y1 + (y2 - y1) * ((a - x1) / (x2 - x1))]
                if midp == endp or midp == startp:
                    break
                if math.fabs(midp[1] - point[1]) > maxerror:
                    output.append(midp)
                    added = True
                    process(series[0:(len(series) / 2) + 1 - n], maxerror, output)
                    process(series[(len(series) / 2) - n:len(series)], maxerror, output)
                else:
                    n += 1
        process(rdata, maxerror, output)
        output.sort(key=lambda x: x[0])
        return output


    def zigzag(self, rdata, threshold):
        """"
        ZIG ZAG RESAMPLING

        Identifies most important features of time series
        based on percentage change of subsequent points
        Takes normalized series and returns resampled series

        Inputs:
               -  list [] of [x, y] float pairs
               -  max error value allowed by compression

        Output:
               - SORTED list [] of [x, y] float pairs
        """
        ndata = []

        for i, x in enumerate(rdata):
            ndata.append([i] + [rdata[i]])

        output = []
        output.append(ndata[0])
        pnt = output[0][1]
        key = 0

        for i, x in enumerate(ndata):

            if (ndata[i][1]-pnt) > threshold and key <= 0:
                key = 1
                output.append([i, ndata[i][1]])
                pnt = ndata[i][1]

            elif (ndata[i][1]-pnt) < threshold*-1 and key >= 0:
                key = -1
                output.append([i, ndata[i][1]])
                pnt = ndata[i][1]

            elif key == 1 and ndata[i][1] > pnt:
                output.insert(-1, [i, ndata[i][1]])
                output.pop(-1)
                pnt = ndata[i][1]

            elif key == -1 and ndata[i][1] < pnt:
                output.insert(-1, [i, ndata[i][1]])
                output.pop(-1)
                pnt = ndata[i][1]

        if len(ndata)-1 != output[-1][0]:
            output.append(ndata[-1])


        return np.transpose(np.asarray(output))
