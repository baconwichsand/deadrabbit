import pdb
import Exchanges
from Strategy import Ogo
from Signal import generateSignals
from Trade import generateTrades
from IOOperations import Config
from pandas import Series, DataFrame
from datetime import datetime

CFG = 'Config/Backtest.cfg'

def backTest(strategy, symbol, start_date, end_date):
    """
    Backtest a strategy
    """

    timing_start = datetime.now()

    ####### LOAD DATA
    data = Exchanges.load_exchange_data(symbol)

    ####### GENERATE SIGNALS
    signals = generateSignals(strategy, data, start_date, end_date)

    ######## LOAD DEFAULTS
    initial_account = float(Config(CFG).get('Defaults', 'initial_account'))

    ######## GENERATE TRADES
    trades, end_account = generateTrades(strategy, symbol, data, signals, initial_account)

    pdb.set_trace()

    ####### GENERATE BACKTEST RESULT
    backtest_labels = ['Symbol', 'Strategy', 'Start Date', 'End Date', 'Account Start', 'Account End', 'Total Return ($)', 'Total Return (%)', 'Total Trades', 'Long Trades', 'Short Trades', 'Stopped-out', 'Execution Win (%)', 'Profit/Trade (%)', 'Loss/Trade (%)', 'Trade Risk/Reward', 'Time/Trade (days)', 'Backtest Speed']

    total_return = end_account - initial_account
    total_return_pct = (end_account / initial_account - 1) * 100
    trades_total = len(trades)
    trades_long = len(trades[trades.Contracts > 0])
    trades_short = len(trades[trades.Contracts < 0])
    stopped_out = len(trades[trades.wasStopped == True])
    execution_win = float(len(trades[((trades.PriceClose-trades.PriceOpen) * (trades.Contracts / trades.Contracts.abs())) > 0])) / float(len(trades))
    trade_pcts = ((trades.PriceClose / trades.PriceOpen - 1) * (trades.Contracts / trades.Contracts.abs()))
    profit_trade = trade_pcts[trade_pcts > 0].mean() * 100
    loss_trade = trade_pcts[trade_pcts < 0].mean() * -100
    risk_reward = profit_trade / loss_trade
    time_trade = (trades.TimestampClose - trades.TimestampOpen).mean()
    speed = (datetime.now() - timing_start)

    return Series([symbol, strategy, start_date, end_date, initial_account, end_account, total_return, total_return_pct, trades_total, trades_long, trades_short, stopped_out, execution_win, profit_trade, loss_trade, risk_reward, time_trade, speed], index=backtest_labels)

SYMBOL = 'NYSE'
STOP_LOSS = 0.025
START_DATE = '1970-01-02'
END_DATE = '2014-11-07'

results = backTest(Ogo(0.025), SYMBOL, START_DATE, END_DATE)
