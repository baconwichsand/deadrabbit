import Indicator
import pdb
from pandas import DataFrame

def generateSignals(strategy, data, start_date, end_date):
    """
    Generates trade signals 
    """
    signals = []

    ####### GET START AND END INDECES
    start_index = data.index.get_loc(start_date)
    end_index = data.index.get_loc(end_date)

    # pdb.set_trace()
    ####### PRECOMPUTE INDICATORS
    indicator_list = strategy.IndicatorList.split(', ')
    for indicator in indicator_list:
        args = [data]
        if '(' in indicator:
            type = indicator.split('(')[0]
            argstring = indicator.split('(')[1].split(')')[0].split(',')
            for arg in argstring:
                args.append(float(arg))
        else:
            type = indicator
        func = getattr(Indicator, type)
        data[indicator] = func(*args)            

    ####### CALCULATE SIGNALS
    i = start_index
    while i < end_index:
        signal = strategy.runLogic(signals, data, i)
        if signal != None:
            signals.append(signal)
        i += 1

    return DataFrame(signals)
