import pdb
from pandas import DataFrame
from pandas.io.data import DataReader
from urllib2 import Request, urlopen
from urllib import urlencode
from IOOperations import Config

CFG = 'Config/Yahoo.cfg'

def get_yticker(symbol):
    """
    Convert symbol to Yahoo Finance Ticker
    """
    return Config(CFG).get("Yahoo Exchange Tickers", 'Y_' + symbol + '_ticker')

def get_daily_history(symbol, start_date, end_date):
    """
    Get historical prices for the given ticker symbol.
    Date format is 'YYYY-MM-DD'

    Returns a nested dictionary (dict of dicts).
    outer dict keys are dates ('YYYY-MM-DD')
    """
    return DataReader(get_yticker(symbol), 'yahoo', start_date, end_date)


SYMBOL = 'NYSE'
START_DATE = '1970-01-01'
END_DATE = '1970-06-01'
rdata = get_daily_history(SYMBOL, START_DATE, END_DATE)
pdb.set_trace()
