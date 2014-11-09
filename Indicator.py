import numpy as np
import h5py
import pdb

def SMA(data, index, period, level='Close'):
    """
    Simple Moving Average (SMA)
    """
    return data['Close'][index+1-period:index+1].mean()


def RANA(data, index):
    """
    Ratio Adjusted Net Advances (RANA)
    """
    return ((data['Advances'] - data['Declines']) / (data['Advances'] + data['Declines']) * 1000)[index]


def McClellanOscillator(data, index):
    """
    McClellan Oscillator

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mcclellan_oscillator
    """
    def EMA19(data, index):
        ema = RANA(data, [index+1-19-19:index+1-19]).mean()
        j = 1
        for i in xrange(19):
            ema = (RANA(data, index-19+j) - ema) * .10 + ema
        return ema

    def EMA39(data, index):
        ema = RANA(data, [index+1-39-39:index+1-39]).mean()
        j = 1
        for i in xrange(39):
            ema = (RANA(data, index-39+j) - ema) * .05 + ema
        return ema
        
    return EMA19(data, index) - EMA39(data, index)


def McClellanSummationIndex(data, index):
    """
    McClellan Summation Index

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mcclellan_summation
    """
    return (McClellanSummationIndex(data, index-1) + McClellanOscillator(data, index))
