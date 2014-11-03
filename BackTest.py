import csv
import numpy as np
import pdb
import ystockquote
from pprint import pprint

f = open('NYSE.csv', 'rt')
reader = csv.reader(f)

### DATE OBJECT = data[:,0]
### ADVANCES = data[:,1]
### DECLINES = data[:,2]
### UNCHANGED = data[:,3]
### RANA = data[:,4]
### p1-day EMA = data[:,5]
### p2-day EMA = data[:,6]
### McClellan Oscillator = data[:,7]
### Summation Index = data[:,8]
### Daily Close = data[:,9]
### Daily Open = 
### 200-d SMA = data[:,10]
data = []
for i, row in enumerate(reader):
    if i != 0:
        data.append(np.array([int(row[0]), int(row[1]), int(row[2]), int(row[3])]))
f.close()
data = np.asarray(data)

### McClellan Oscillator Calculation
### http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mcclellan_oscillator

### RANA
rana = np.array([float((d[1]-d[2])/(d[1]+d[2]))*1000 for d in data])
data = np.column_stack((data, rana))

ema19 = np.repeat(-1, 19)
ema19 = np.append(ema19, np.mean(rana[:19]))
for i, r in enumerate(rana):
    if i > 19:
        tmp = (r - ema19[-1]) * .10 + ema19[-1]
        ema19 = np.append(ema19, tmp)
data = np.column_stack((data, ema19))

ema39 = np.repeat(-1, 39)
ema39 = np.append(ema39, np.mean(data[:,4][:39]))
for i, r in enumerate(data[:,4]):
    if i > 39:
        tmp = (r - ema39[-1]) * .05 + ema39[-1]
        ema39 = np.append(ema39, tmp)
data = np.column_stack((data, ema39))

mosc = np.array([d[5]-d[6] for d in data])
data = np.column_stack((data, mosc))

summind = np.repeat(-1, 39)
summind = np.append(summind, data[:,7][39])
for i, r in enumerate(rana):
    if i > 39:
        tmp = data[:,7][i] + summind[-1]
        summind = np.append(summind, tmp)
data = np.column_stack((data, summind))

### PULL YAHOO FINANCE DATA
def convert_to_float(string):
    return float(str(string[:4] + string[5:7] + string[8:10]))

def convert_to_string(flt):
    return str(int(flt))[:4] + '-' + str(int(flt))[4:6] + '-' + str(int(flt))[6:8]

start_date = '2000-01-03'
end_date = '2014-10-22'
ticker = '^NYA'

ydata = ystockquote.get_historical_prices(ticker, start_date, end_date)

data = data[np.where(data[:,0] >= convert_to_float(start_date))]
data = data[np.where(data[:,0] <= convert_to_float(end_date))]

count = 0
for date in data[:,0]:
    if convert_to_string(date) not in ydata:
        count += 1
        print date
print count

close_data = []
for i, row in enumerate(data):
    if convert_to_string(row[0]) in ydata:
        close = float(ydata[convert_to_string(row[0])]['Close'])
    else:
        close = close_data[i-1]
    close_data.append(close)

data = np.column_stack((data, close_data))

### 200-d SMA
sma = []
for i, date in enumerate(data):
    sample = []
    while len(sample) < 201:
        sample.append(date[9])
    if len(sample) == 200:
        sample = sample[1:]
        sample.append(date[9])
    sma.append(np.mean(sample))

data = np.column_stack((data, sma))

### START BACKTEST

long_trade_active = False
short_trade_active = False

entry_long = 0
entry_short = 0

longs = 0
shorts = 0
results = []

days_long_entry = []
days_long_close = []
days_short_entry = []
days_short_close = []

csv_output = []

for i, day in enumerate(data):

    if i > 2:

        if ((data[i-2][8] > data[i-1][8]) and (data[i-1][8] > data[i][8])) and (data[i][9] > data[i][10]) and not long_trade_active:

            longs += 1
            long_trade_active = True
            entry_long = data[i][9]
            days_long_entry.append(data[i][0])

        if ((data[i-2][8] < data[i-1][8]) and (data[i-1][8] < data[i][8])) and (data[i][9] > data[i][10]) and long_trade_active:

            long_trade_active = False
            days_long_close.append(data[i][0])

            if entry_long < data[i][9]:
                results.append(1)
            else:
                results.append(0)

        if ((data[i-2][8] < data[i-1][8]) and (data[i-1][8] < data[i][8])) and (data[i][9] < data[i][10]) and not short_trade_active:

            shorts += 1
            short_trade_active = True
            entry_short = data[i][9]
            days_short_entry.append(data[i][0])

        if ((data[i-2][8] > data[i-1][8]) and (data[i-1][8] > data[i][8])) and (data[i][9] < data[i][10]) and long_trade_active:

            short_trade_active = False
            days_short_close.append(data[i][0])

            if entry_short > data[i][9]:
                results.append(1)
            else:
                results.append(0)

print float(sum(results))/float(len(results))
print longs
print shorts
pdb.set_trace()
