import numpy as np
from pandas.stats.moments import ewma
import pandas as pd
import h5py
import pdb

def SMA(data, period, level='Close'):
    """
    Simple Moving Average (SMA) for entire dataset
    """
    return pd.rolling_mean(data[level], period)

def McClellanSummationIndex(data):
    """
    McClellan Summation Index for entire dataset
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mcclellan_summation
    """
    def McClellanOscillator():
        """
        McClellan Oscillator
        http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mcclellan_oscillator
        """
        def RANA():
            """
            Ratio Adjusted Net Advances (RANA)
            Returns RANA for whole dataset
            """
            return ((data['Advances'] - data['Declines']) / (data['Advances'] + data['Declines']) * 1000)
        ####### Calculate RANA
        rana = RANA()
        ####### Calculate EMAS
        ema19 = ewma(rana, span=19)
        ema39 = ewma(rana, span=39)
        return ema19 - ema39
    ####### Calculate Oscillator
    oscillator = McClellanOscillator()
    ####### Calculate Summations
    summations = oscillator.cumsum()
    return summations

def RSI(data, period=14):
    """
    Relative Strength Index
    http://www.investopedia.com/terms/r/rsi.asp
    """
    def rsiCalc(p):
        # subfunction for calculating rsi for one lookback period
        avgGain = p[p>0].sum()/period
        avgLoss = -p[p<0].sum()/period
        rs = avgGain/avgLoss
        return 100 - 100/(1+rs)

    gain = (data.Close - data.Close.shift(1)).fillna(0)

    # run for all periods with rolling_apply
    return pd.rolling_apply(gain,period,rsiCalc) 
