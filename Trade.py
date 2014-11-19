from pandas import DataFrame, Series
import inspect
import pdb
import numpy as np

# def newTrade(symbol, timestamp_open, order_open, order_close=None, timestamp_close=None, backtest=True):

class NS:
    pass

def tradeDataLabels():
    return ['isOpen', 'Symbol', 'Contracts', 'TimestampOpen', 'PriceOpen', 'TimestampClose', 'PriceClose', 'StopLoss', 'wasStopped']


def generateTrades(strategy, symbol, data, signals, account):

    NS.trades = DataFrame(columns=tradeDataLabels())

    def openTrade(tsp, price, sl, contracts):
        NS.trades = NS.trades.append(Series([True, symbol, contracts, tsp, price, None, None, sl, False], index=tradeDataLabels()), ignore_index=True)

    def closeTrades(indeces, tsp, price, stopped=False):
        NS.trades.ix[indeces, np.where(NS.trades.columns.values == 'TimestampClose')[0][0]] = tsp
        NS.trades.ix[indeces, np.where(NS.trades.columns.values == 'PriceClose')[0][0]] = price
        NS.trades.ix[indeces, np.where(NS.trades.columns.values == 'wasStopped')[0][0]] = stopped
        NS.trades.ix[indeces, np.where(NS.trades.columns.values == 'isOpen')[0][0]] = False
        return NS.trades.ix[indeces]

    for i in range(len(data)):

        active_trades = NS.trades[NS.trades.isOpen == True]
        active_long_trades = active_trades[active_trades.Contracts > 0]
        ############## TODO : ADD SITUATION WHERE NO STOP CURRENTLY NOT FUNCTIONING
        active_long_trades_sl = active_long_trades
        active_short_trades = active_trades[active_trades.Contracts < 0]
        active_short_trades_sl = active_short_trades

        ####### IF STOP HIT ON LONG TRADE
        if not active_long_trades_sl[active_long_trades_sl.StopLoss >= data.ix[i].Close].empty:
            indeces = active_long_trades_sl[active_long_trades_sl.StopLoss >= data.ix[i].Close].index
            trade = closeTrades(indeces, data.ix[i].name, data.ix[i].Close, stopped=True)
            gain_nominal = (trade.PriceClose.values[0] - trade.PriceOpen.values[0]) * trade.Contracts.values[0]
            account = account + gain_nominal

        ####### IF STOP HIT ON SHORT TRADE
        if not active_short_trades_sl[active_short_trades_sl.StopLoss <= data.ix[i].Close].empty:
            indeces = active_short_trades_sl[active_short_trades_sl.StopLoss <= data.ix[i].Close].index
            trade = closeTrades(indeces, data.ix[i].name, data.ix[i].Close, stopped=True)
            gain_nominal = (trade.PriceClose.values[0] - trade.PriceOpen.values[0]) * trade.Contracts.values[0]
            account = account + gain_nominal

        ####### CHECK FOR SIGNAL
        if not signals[signals.date == data.ix[i].name].empty:
            signal_tsp = signals[signals.date == data.ix[i].name].date.values[0]
            signal_type = signals[signals.date == data.ix[i].name].type.values[0]

            if signal_type == 'buy_to_open':
                start_account = account
                price = data.ix[i].Close
                sl = data.ix[i].Close * (1 - strategy.stop_loss)
                contracts = start_account / data.ix[i].Close
                openTrade(signal_tsp, price, sl, contracts)

            elif signal_type == 'sell_to_close':
                if not active_long_trades.empty:
                    trade = closeTrades(active_long_trades.index, signal_tsp, data.ix[i].Close)
                    gain_nominal = (trade.PriceClose.values[0] - trade.PriceOpen.values[0]) * trade.Contracts.values[0]
                    account = account + gain_nominal

            elif signal_type == 'sell_to_open':
                start_account = account
                price = data.ix[i].Close
                sl = data.ix[i].Close * (1 - strategy.stop_loss)
                contracts = (start_account / data.ix[i].Close) * -1
                openTrade(signal_tsp, price, sl, contracts)

            elif signal_type == 'buy_to_close':
                if not active_short_trades.empty:
                    trade = closeTrades(active_short_trades.index, signal_tsp, data.ix[i].Close)
                    gain_nominal = (trade.PriceClose.values[0] - trade.PriceOpen.values[0]) * trade.Contracts.values[0]
                    account = account + gain_nominal

    return NS.trades, account
