"""
Version 1
Module that calculates differnce varius score between equal length arrays

Arguments:
    trng_set - Training data subset
    trgt_set - Target data subset
    mtch_args - (Base, Exponent, Lower Limit, Upper Limit)

"""
import numpy as np
def cum_score_01(list_):
    """
    Calculates cumulative matching score,
       precision is not tested on large bset's.

    Argument: matching_01 output ( array(1, n) )
    """
    output = (1 - np.sum(list_)/len(list_))

    if output < 0:
        return 0.0

    return output



def matching_01(trng_bset, trgt_bset, mtch_args):
    """
    Main function for matching_01
    not implemented local limits and not optimized.

    """
    def error_score(trng_n, trgt_n):
        """Calculates error score element wise"""

        # Calculate Differnce between model element and target data subset
        diff = trng_n - trgt_n
        #print "df", diff

        # Check which error calculation limit to use (upper or lower)
        if diff < 0:
            err_prcnt = 1 - (abs(diff)/mtch_args[2]) # Lower Limit
            # print "er1", err_prcnt
        else:
            err_prcnt = 1 - (abs(diff)/mtch_args[3]) # Upper Limit
            # print "er2", err_prcnt

        # base mtch_args[0]
        # exponent mtch_args[1]
        if err_prcnt > 0:
            #print "n0", err_prcnt
            return (mtch_args[0]**((2-err_prcnt)**mtch_args[1])\
                            - mtch_args[0]) / (mtch_args[0]**(2**mtch_args[1])\
                            - mtch_args[0])
        else:
            #print "n1", 1 + abs(err_prcnt)
            return 1 + abs(err_prcnt)

    list_ = []

    for trng, trgt in zip(trng_bset, trgt_bset):
        list_.append(error_score(trng, trgt))

    #print list_
    return list_


def matching_02(trang_set, targt_set, args_group):
    """
    Main function for matching_01
    not implemented local limits and not optimized.

    """
    diff_5 = [0, 0]

    def calc_prx(err_prx, base, expo):
        if err_prx > 0:
            out = (base**((2-err_prx)**expo) - base) / (base**(2**expo) - base)
            return out
        else:
            return 1 + err_prx*-1

    vfunc = np.vectorize(calc_prx)

    diff_1 = np.subtract(trang_set, targt_set)
    diff_x = np.arange(len(diff_1))
    diff_1a = np.absolute(diff_1)


    diff_11 = np.divide(diff_1a, args_group[2])
    diff_12 = np.divide(diff_1a, args_group[3])

    diff_0 = diff_11[np.where(diff_1 < 0)]
    diff_0x = diff_x[np.where(diff_1 < 0)]

    diff_01 = diff_11[np.where(diff_1 >= 0)]
    diff_01x = diff_x[np.where(diff_1 >= 0)]

    error_prx0 = np.subtract(1, diff_0)
    error_prx1 = np.subtract(1, diff_01)

    error_prx_y = np.hstack((error_prx0, error_prx1))
    error_prx_x = np.hstack((diff_0x, diff_01x))

    a = vfunc(error_prx_y, 3 ,2)

    error_prx = np.vstack((error_prx_x, a))

    return error_prx[1]



# import mlpy
# import rand_generators
# import DataManipulation as dm

# rdata1 = dm.normalize(rand_generators.rand_vol(50, 1000, 0.03))
# rdata1 = dm.normalize(rand_generators.rand_vol(80, 1000, 0.03))
