from pandas import Series, DataFrame
from IOOperations import Config
from urllib2 import Request, urlopen

CFG = 'Config/Internals.cfg'

def update_unicorn(symbol):

    adv = []
    decl = []
    unch = []
    time = []
    advances_data = urlopen(Config(CFG).get(symbol, 'advances'))
    declines_data = urlopen(Config(CFG).get(symbol, 'declines'))
    unchanged_data = urlopen(Config(CFG).get(symbol, 'unchanged'))

    for line1, line2, line3 in zip(advances_data, declines_data, unchanged_data):
        tm = line1.split(', ')[0]
        time.append(tm[:4] + '-' + tm[4:6] + '-' + tm[6:8])
        adv.append(float(line1.split(', ')[1][:-1]))
        decl.append(float(line2.split(', ')[1][:-1]))
        unch.append(float(line3.split(', ')[1][:-1]))

    output = DataFrame({'Advances' : adv, 'Declines' : decl, 'Unchanged': unch}, index=time)
    return output

   
# SYMBOL = 'NYSE'
# START_DATE = '1970-05-23'
# END_DATE = '2000-05-23'

# rdata = update_unicorn(SYMBOL, START_DATE, END_DATE)
