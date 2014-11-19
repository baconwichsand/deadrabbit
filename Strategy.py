import pdb

class Ogo:

    def __init__(self, stoploss):

        self.stop_loss = stoploss

    def runLogic(self, signals, data, i):

        ####### IF (CLOSE > 200 DAY SMA) AND (RSI)
        if (data.ix[i].Close > data.ix[i]['SMA(200)']) and (data.ix[i]['RSI(14)'] > 50):

           ####### IF 2 DAYS OF INCREASING SUMMATION INDEX AFTER TURN
            if (data.ix[i-3].McClellanSummationIndex > data.ix[i-2].McClellanSummationIndex) and (data.ix[i-2].McClellanSummationIndex < data.ix[i-3].McClellanSummationIndex) and (data.ix[i-1].McClellanSummationIndex < data.ix[i].McClellanSummationIndex):

                ####### BUY TO OPEN
                return {'type':'buy_to_open', 'date':data.index[i]}

            ####### IF 2 DAYS OF DECREASING SUMMATION INDEX AFTER TURN
            if (data.ix[i-3].McClellanSummationIndex < data.ix[i-2].McClellanSummationIndex) and (data.ix[i-2].McClellanSummationIndex > data.ix[i-3].McClellanSummationIndex) and (data.ix[i-1].McClellanSummationIndex > data.ix[i].McClellanSummationIndex):

                ####### SELL TO CLOSE
                return {'type':'sell_to_close', 'date':data.index[i]}

        ####### IF CLOSE < 200 DAY SMA
        if (data.ix[i].Close < data.ix[i]['SMA(200)']) and (data.ix[i]['RSI(14)'] < 50):

            ####### IF 2 DAYS OF DECREASING SUMMATION INDEX AFTER TURN
            if (data.ix[i-3].McClellanSummationIndex < data.ix[i-2].McClellanSummationIndex) and (data.ix[i-2].McClellanSummationIndex > data.ix[i-3].McClellanSummationIndex) and (data.ix[i-1].McClellanSummationIndex > data.ix[i].McClellanSummationIndex):

                ####### SELL TO OPEN
                return {'type':'sell_to_open', 'date':data.index[i]}

            ####### IF 2 DAYS OF INCREASING SUMMATION INDEX AFTER TURN
            if (data.ix[i-3].McClellanSummationIndex > data.ix[i-2].McClellanSummationIndex) and (data.ix[i-2].McClellanSummationIndex < data.ix[i-3].McClellanSummationIndex) and (data.ix[i-1].McClellanSummationIndex < data.ix[i].McClellanSummationIndex):

                ####### SELL TO OPEN
                return {'type':'buy_to_close', 'date':data.index[i]}

        return None

    @property
    def IndicatorList(self):
        return "SMA(200), McClellanSummationIndex, RSI(14)"
