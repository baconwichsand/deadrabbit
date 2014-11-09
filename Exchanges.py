from IOOperations import Config
from datetime import datetime
from SysUtil import file_update_backup
from Internals import update_unicorn
from pandas.io.pytables import HDFStore
from Yahoo import get_daily_history

CFG = 'Config/Exchanges.cfg'


def load_exchange_data(symbol):
    """
    Returns data for a specific exchange

    """
    filename = Config(CFG).get("DB Locations", 'exchange_data')
    operator = HDFStore(filename)
    return operator[symbol]


def update_exchanges():
    """
    Updates data for exchanges such as NYSE

    """

    ####### LOAD DATE RANGES AND SYMBOLS
    start_date = Config(CFG).get('Exchange Data Start Date', 'default_start_date')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    symbols = [Config(CFG).get('Symbol List', 'list')]
    ####### BACKUP and UPDATE DB
    filename = Config(CFG).get("DB Locations", 'exchange_data')
    backup = Config(CFG).get("DB Locations", 'exchange_data_backup')
    file_update_backup(filename, backup)
    ####### START HDF5 INSTANCE
    operator = HDFStore(filename)

    for symbol in symbols:

        ####### PULL YAHOO FINANCE DATA
        data = get_daily_history(symbol, start_date, end_date)
        ####### PULL ADVANCES/DECLINES DATA
        data = data.merge(update_unicorn(symbol), left_index=True, right_index=True, how='outer')
        ####### SAVE DATA TO HDF5
        operator[symbol] = data

    operator.close()





# SYMBOL = 'NYSE'
# START_DATE = '1965-03-01'
# END_DATE = '2000-05-23'

# rdata = update_exchanges()
