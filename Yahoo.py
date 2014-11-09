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

    # params = urlencode({
    #     's': get_yticker(symbol),
    #     'a': int(start_date[5:b7]) - 1,
    #     'b': int(start_date[8:10]),
    #     'c': int(start_date[0:4]),
    #     'd': int(end_date[5:7]) - 1,
    #     'e': int(end_date[8:10]),
    #     'f': int(end_date[0:4]),
    #     'g': 'd',
    #     'ignore': '.csv',
    # })
    # url = 'http://ichart.yahoo.com/table.csv?%s' % params
    # req = Request(url)
    # resp = urlopen(req)
    # content = str(resp.read().decode('utf-8').strip())
    # daily_data = content.splitlines()
    # hist_dict = dict()
    # keys = daily_data[0].split(',')
    # for day in daily_data[1:]:
    #     day_data = day.split(',')
    #     date = day_data[0]
    #     hist_dict[date] = \
    #         {keys[1]: day_data[1],
    #          keys[2]: day_data[2],
    #          keys[3]: day_data[3],
    #          keys[4]: day_data[4],
    #          keys[5]: day_data[5],
    #          keys[6]: day_data[6]}
    # return DataFrame(hist_dict).T

# SYMBOL = 'NYSE'
# START_DATE = '1970-01-01'
# END_DATE = '1970-06-01'

# rdata = get_daily_history(SYMBOL, START_DATE, END_DATE)
# pdb.set_trace()
