import pdb
import sys
import numpy as np
import csv
import urllib2
import h5py
import cmd
import time
import datetime
import pylab as plt
import Exchanges
from matplotlib import rc
from matplotlib.widgets import Slider, Button, RadioButtons
from Indicator import RANA, SMA
from InputBuilder import TargetData, TrainingData, Results
from io_operations import Config
from Visualization import feature_selection, matching_review
import os.path
from copy import deepcopy
from DataManipulation import normalize, eq_scale
import matching
from analysis import TrainingAnalysis
from progressbar import Bar, SimpleProgress, ReverseBar, ProgressBar


# ##### SMOOTH HDF5 EXECUTION #####
sys.dont_write_bytecode = True

# #### LOAD CONFIG FILE TO START SESSION

CFG = 'Config/DeadRabbit.cfg'
DB = Config(CFG).get("DB Locations", 'test')

class NS:
    pass

class DRconsole(cmd.Cmd):

    ####################################################
    # UPDATE EXCHANGES
    ####################################################
    def do_update_exchanges(self, line):

        Exchanges.update_exchanges()

    ####################################################
    # RUN GENERATE SIGNALS
    ####################################################
    def do_generate_signals(self, line):

        ####### GET ARGUMENTS
        exchange = raw_input('Enter Exchange: ')
        while exchange == '':
            print 'Exchange symbol required'
            exchange = raw_input('Enter Exchange: ')
        start_date = raw_input('Enter start date (default 2000-01-01): ')
        if start_date == '':
            start_date = '1970-01-04'
        end_date = raw_input('Enter end date (default last close): ')
        if end_date == '':
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        ####### LOAD DATA
        data = Exchanges.load_exchange_data(exchange)


    ####################################################
    # VIEW SIGNALS
    ####################################################
    def do_view_signals(self, line):

        ####### GET ARGUMENTS
        exchange = raw_input('Enter Exchange: ')
        while exchange == '':
            print 'Exchange symbol required'
            exchange = raw_input('Enter Exchange: ')
        signals_name = raw_input('Enter signals name: ')
        while signals_name == '':
            exchange = raw_input('Enter signals name: ')
        
        ####### LOAD DATA
        filename = 'data/' + exchange + '.hdf5'
        operator = h5py.File(filename, 'a')
        data = operator[exchange][...]
        signals_dset = operator['/Signals/' + signals_name]
        signals = signals_dset[...]

        ####### TIMEFRAME
        start_date = signals_dset.attrs['start_date']
        start_date_float = float(start_date.split('-')[0] + start_date.split('-')[1] + start_date.split('-')[2])
        end_date = signals_dset.attrs['end_date']
        end_date_float = float(end_date.split('-')[0] + end_date.split('-')[1] + end_date.split('-')[2])
        data = data[np.where(data[:,0] >= start_date_float)]
        data = data[np.where(data[:,0] <= end_date_float)]

        fig, (plt1, plt2, plt3) = plt.subplots(3, 1)
        frame = 200

        ####### INITIAL FRAME
        NS.signal_index = 0
        NS.data_index = np.where(data[:,0] == signals[NS.signal_index][0])[0][0]
        if NS.data_index - frame > 0:
            frame_start = NS.data_index - frame
        else:
            frame_start = 0
        if NS.data_index + frame > len(data):
            frame_end = len(data)
        else:
            frame_end = NS.data_index + frame

        plt1.grid(True)
        plt1.hold(True)
        plt1.plot(data[:,4][frame_start:frame_end], color='c')
        plt1.plot(data[:, 15][frame_start:frame_end], color='r')
        plt1.plot([frame, frame], [min(data[:,4][frame_start:frame_end]), max(data[:,4][frame_start:frame_end])], 'k', lw=1)

        plt2.grid(True)
        plt2.plot(data[:, 14][frame_start:frame_end], color='g')
        plt2.plot([frame, frame], [min(data[:,14][frame_start:frame_end]), max(data[:,14][frame_start:frame_end])], 'k', lw=1)

        symbol_string = []
        if signals[NS.signal_index][1] == 0:
            symbol_string.append('Buy to Open')
        if signals[NS.signal_index][1] == 1:
            symbol_string.append('Sell to Open')
        if signals[NS.signal_index][1] == 2:
            symbol_string.append('Sell to Close')
        if signals[NS.signal_index][1] == 3:
            symbol_string.append('Buy to Close')

        date_float = data[:,0][NS.data_index]
        date_str = str(date_float)[:4] + '-' + str(date_float)[4:6] + '-' + str(date_float)[6:8]
        symbol_string.append(date_str)
        symbol_string.append(str(data[:,4][NS.data_index]))
        symbol_string.append(str(data[:,15][NS.data_index]))
        symbol_string.append(str(data[:,14][NS.data_index]))

        rc('font', family='serif')
        rc('text', usetex=True)

        plt3.cla()
        plt3.text(0.5, 0.5, r'\begin{tabular}{lc} {Signal: } & {%s} \\ {Date: } & {%s} \\ {Close: } & {%s} \\ {SMA: } & {%s} \\ {Summation Index: } & {%s} \\ \end{tabular}' % tuple(symbol_string), fontsize=12)


        axcol = 'lightgoldenrodyellow'

        ##### BUILD NEXT SIGNAL BUTTON
        nextax = plt.axes([0.57, 0.03, 0.065, 0.035])
        nextbutton = Button(nextax, 'Next Signal', color=axcol, hovercolor='0.0975')
        def next_signal(event):
            NS.signal_index += 1
            NS.data_index = np.where(data[:,0] == signals[NS.signal_index][0])[0][0]

            if (NS.data_index - frame) > 0:
                frame_start = NS.data_index - frame
            else:
                frame_start = 0
            if (NS.data_index + frame) > len(data):
                frame_end = len(data)
            else:
                frame_end = NS.data_index + frame

            plt1.cla()
            plt1.grid(True)
            plt1.hold(True)
            plt1.plot(data[:,4][frame_start:frame_end], color='c')
            plt1.plot(data[:, 15][frame_start:frame_end], color='r')
            plt1.plot([frame, frame], [min(data[:,4][frame_start:frame_end]), max(data[:,4][frame_start:frame_end])], 'k', lw=1)

            plt2.cla()
            plt2.grid(True)
            plt2.plot(data[:, 14][frame_start:frame_end], color='g')
            plt2.plot([frame, frame], [min(data[:,14][frame_start:frame_end]), max(data[:,14][frame_start:frame_end])], 'k', lw=1)

            symbol_string = []
            if signals[NS.signal_index][1] == 0:
                symbol_string.append('Buy to Open')
            if signals[NS.signal_index][1] == 1:
                symbol_string.append('Sell to Open')
            if signals[NS.signal_index][1] == 2:
                symbol_string.append('Sell to Close')
            if signals[NS.signal_index][1] == 3:
                symbol_string.append('Buy to Close')

            date_float = data[:,0][NS.data_index]
            date_str = str(date_float)[:4] + '-' + str(date_float)[4:6] + '-' + str(date_float)[6:8]
            symbol_string.append(date_str)
            symbol_string.append(str(data[:,4][NS.data_index]))
            symbol_string.append(str(data[:,15][NS.data_index]))
            symbol_string.append(str(data[:,14][NS.data_index]))

            rc('font', family='serif')
            rc('text', usetex=True)

            plt3.cla()
            plt3.text(0.5, 0.5, r'\begin{tabular}{lc} {Signal: } & {%s} \\ {Date: } & {%s} \\ {Close: } & {%s} \\ {SMA: } & {%s} \\ {Summation Index: } & {%s} \\ \end{tabular}' % tuple(symbol_string), fontsize=12)
            plt.draw()
        nextbutton.on_clicked(next_signal)

        ##### BUILD PREVIOUS SIGNAL BUTTON
        prevax = plt.axes([0.47, 0.03, 0.065, 0.035])
        prevbutton = Button(prevax, 'Previous Signal', color=axcol, hovercolor='0.0975')
        def prev_signal(event):
            NS.signal_index -= 1
            NS.data_index = np.where(data[:,0] == signals[NS.signal_index][0])[0][0]

            if (NS.data_index - frame) > 0:
                frame_start = NS.data_index - frame
            else:
                frame_start = 0
            if (NS.data_index + frame) > len(data):
                frame_end = len(data)
            else:
                frame_end = NS.data_index + frame

            plt1.cla()
            plt1.grid(True)
            plt1.hold(True)
            plt1.plot(data[:,4][frame_start:frame_end], color='c')
            plt1.plot(data[:, 15][frame_start:frame_end], color='r')
            plt1.plot([frame, frame], [min(data[:,4][frame_start:frame_end]), max(data[:,4][frame_start:frame_end])], 'k', lw=1)

            plt2.cla()
            plt2.grid(True)
            plt2.plot(data[:, 14][frame_start:frame_end], color='g')
            plt2.plot([frame, frame], [min(data[:,14][frame_start:frame_end]), max(data[:,14][frame_start:frame_end])], 'k', lw=1)

            symbol_string = []
            if signals[NS.signal_index][1] == 0:
                symbol_string.append('Buy to Open')
            if signals[NS.signal_index][1] == 1:
                symbol_string.append('Sell to Open')
            if signals[NS.signal_index][1] == 2:
                symbol_string.append('Sell to Close')
            if signals[NS.signal_index][1] == 3:
                symbol_string.append('Buy to Close')

            date_float = data[:,0][NS.data_index]
            date_str = str(date_float)[:4] + '-' + str(date_float)[4:6] + '-' + str(date_float)[6:8]
            symbol_string.append(date_str)
            symbol_string.append(str(data[:,4][NS.data_index]))
            symbol_string.append(str(data[:,15][NS.data_index]))
            symbol_string.append(str(data[:,14][NS.data_index]))

            rc('font', family='serif')
            rc('text', usetex=True)

            plt3.cla()
            plt3.text(0.5, 0.5, r'\begin{tabular}{lc} {Signal: } & {%s} \\ {Date: } & {%s} \\ {Close: } & {%s} \\ {SMA: } & {%s} \\ {Summation Index: } & {%s} \\ \end{tabular}' % tuple(symbol_string), fontsize=12)
            plt.draw()
        prevbutton.on_clicked(prev_signal)


        ##### BUILD LAST SIGNAL BUTTON
        lastax = plt.axes([0.37, 0.03, 0.065, 0.035])
        lastbutton = Button(lastax, 'Last Signal', color=axcol, hovercolor='0.0975')
        def last_signal(event):
            NS.signal_index -= 1
            NS.data_index = np.where(data[:,0] == signals[NS.signal_index][0])[0][0]

            if (NS.data_index - frame) > 0:
                frame_start = NS.data_index - frame
            else:
                frame_start = 0
            if (NS.data_index + frame) > len(data):
                frame_end = len(data)
            else:
                frame_end = NS.data_index + frame

            plt1.cla()
            plt1.grid(True)
            plt1.hold(True)
            plt1.plot(data[:,4][frame_start:frame_end], color='c')
            plt1.plot(data[:, 15][frame_start:frame_end], color='r')
            plt1.plot([frame, frame], [min(data[:,4][frame_start:frame_end]), max(data[:,4][frame_start:frame_end])], 'k', lw=1)

            plt2.cla()
            plt2.grid(True)
            plt2.plot(data[:, 14][frame_start:frame_end], color='g')
            plt2.plot([frame, frame], [min(data[:,14][frame_start:frame_end]), max(data[:,14][frame_start:frame_end])], 'k', lw=1)

            symbol_string = []
            if signals[NS.signal_index][1] == 0:
                symbol_string.append('Buy to Open')
            if signals[NS.signal_index][1] == 1:
                symbol_string.append('Sell to Open')
            if signals[NS.signal_index][1] == 2:
                symbol_string.append('Sell to Close')
            if signals[NS.signal_index][1] == 3:
                symbol_string.append('Buy to Close')

            date_float = data[:,0][NS.data_index]
            date_str = str(date_float)[:4] + '-' + str(date_float)[4:6] + '-' + str(date_float)[6:8]
            symbol_string.append(date_str)
            symbol_string.append(str(data[:,4][NS.data_index]))
            symbol_string.append(str(data[:,15][NS.data_index]))
            symbol_string.append(str(data[:,14][NS.data_index]))

            rc('font', family='serif')
            rc('text', usetex=True)

            plt3.cla()
            plt3.text(0.5, 0.5, r'\begin{tabular}{lc} {Signal: } & {%s} \\ {Date: } & {%s} \\ {Close: } & {%s} \\ {SMA: } & {%s} \\ {Summation Index: } & {%s} \\ \end{tabular}' % tuple(symbol_string), fontsize=12)
            plt.draw()
        lastbutton.on_clicked(last_signal)


        ##### BUILD EXIT BUTTON
        exitax = plt.axes([0.83, 0.03, 0.065, 0.035])
        exitbutton = Button(exitax, 'Exit', color=axcol, hovercolor='0.0975')
        def leave(event):
            plt.close()
        exitbutton.on_clicked(leave)

        plt.draw()
        plt.show()

    ####################################################
    # RUN BACKTEST W/ SIGNALS
    ####################################################
    def do_backtest(self, line):
        

        ####### GET ARGUMENTS
        exchange = raw_input('Enter Exchange: ')
        while exchange == '':
            print 'Exchange symbol required'
            exchange = raw_input('Enter Exchange: ')
        signals_name = raw_input('Enter signals name: ')
        while signals_name == '':
            exchange = raw_input('Enter signals name: ')
        
        ####### LOAD DATA
        filename = 'data/' + exchange + '.hdf5'
        operator = h5py.File(filename, 'a')
        symbol_data = operator[exchange][...]
        signals = operator['/Signals/' + signals_name]


        


        ####### SAVE DATA
        results_name = raw_input('Enter backtest name: ')
        operator[results_name] = signals
        dset = operator[results_name]
        dset.attrs['start_date'] = start_date
        dset.attrs['end_date'] = end_date
        dset.attrs['exchange'] = exchange
        operator.close()


    ####################################################
    # GENERATE TARGET DATA
    ####################################################
    def do_generate_target_data(self, line):
        tdata_builder = TargetData(DB)

        correct_input = False
        while not correct_input:
            data = raw_input('Generate random data or load from file? (r/f) ')
            if data == 'r':
                correct_input = True
                generator = raw_input('Randgen type (volatility (v) / range (r)): ')
                if generator == 'r':
                    gentype = 'range'
                    min_int = int(raw_input('Min int: '))
                    max_int = int(raw_input('Max int: '))
                    params = [min_int, max_int]
                if generator == 'v':
                    gentype = 'volatility'
                    start_value = int(raw_input('Start value (int): '))
                    volatility = float(raw_input('Volatility: '))
                    params = [start_value, volatility]
                num_points = int(raw_input('Number of points: '))
                name = raw_input('Give dataset name: ')
                tdata_builder.random(name, num_points, gentype, params)
                print '====> Dataset ' + name + ' generated and stored'
            elif data == 'f':
                correct_input = True
                series = raw_input('Filename (must be .csv): ')
                if not os.path.isfile(series):
                    loaded = False
                    while not loaded:
                        series = raw_input(series + ' not found. Try again: ')
                        if os.path.isfile(series):
                            loaded = True
                name = raw_input('Give dataset name: ')
                tdata_builder.csv(series, name)
                print '====> Dataset ' + name + 'generated and stored'
            else:
                print '\'' + data + '\'' + ' not a valid input'

    ####################################################
    #   GENERATE TRAINING DATA
    ####################################################
    def do_generate_training_data(self, line):

        # #### LOAD DATA
        tdata_builder = TargetData(DB)
        name = raw_input('Load dataset (enter name): ')
        rdata = tdata_builder.load(name)

        # ##### READ PARAMETERS FROM CONFIG
        config = Config(CFG)
        params = [config.get('Feature Identification', 'subset_length', \
                             dtype=int),
                  config.get('Feature Identification', 'subset_start', \
                             dtype=int),
                  config.get('Feature Identification', 'zigzag', dtype=float), \
                  config.get('Feature Identification', 'fractal', dtype=float)]
        print '====> Feature selection params loaded'

        # ##### START INTERACTIVE FEATURE SELECTION
        training_data = feature_selection(params, rdata)

        # ##### STORE TRAINING DATA IN DATABASE
        name = raw_input('Give dataset name: ')
        TrainingData(DB).save(training_data, name)

    ####################################################
    #   CALCULATE MATCHING SCORE
    ####################################################
    def do_matching_score(self, line):

        # #### LOAD TARGET DATA
        target_data_builder = TargetData(DB)
        target_data_name = raw_input('Load Target Data (enter name): ')
        target_data = target_data_builder.load(target_data_name)

        # #### LOAD TRAINING DATA
        training_data_builder = TrainingData(DB)
        training_data_name = raw_input('Load Training Data (enter name): ')
        training_data = training_data_builder.load(training_data_name)

        # ##### READ PARAMETERS FROM CONFIG
        config = Config(CFG)
        base = config.get('Matching', 'base', dtype=int)
        exponent = config.get('Matching', 'exponent', dtype=int)
        gbl_upper_limit = config.get('Matching', 'gbl_lower_limit', dtype=float)
        gbl_lower_limit = config.get('Matching', 'gbl_lower_limit', dtype=float)
        params = [base, exponent, gbl_upper_limit, gbl_lower_limit]
        print '====> Feature selection params loaded'

        subset_index_array = []
        rng_current = []
        min_length = 100
        max_length = 500
        range_step_length = 100
        step_through_length = 100
        for i in xrange(min_length, max_length, range_step_length):
            rng_current = [0, i]
            while (rng_current[1]) < len(target_data)/2:
                subset_index_array.append(deepcopy(rng_current))
                rng_current[0] += step_through_length
                rng_current[1] += step_through_length

        results = []
        n = 0
        widgets = [Bar('>'), ' ', SimpleProgress(), ' ', ReverseBar('<')]
        pbar = ProgressBar(widgets=widgets, maxval=len(subset_index_array)*len(training_data)).start()
        for i, x in enumerate(subset_index_array):
            subset = target_data[x[0]:x[1]]
            subset = normalize(subset)
            for j, y in enumerate(training_data):

                ##### OLD MATCHING
                equalized = eq_scale(y, subset)
                m = matching.cum_score_01(matching.matching_01(equalized[0], equalized[1], params))
                # ##### DTW MATCHING
                # m = mlpy.dtw_std(y, subset, dist_only=True)
                results.append([x[0], x[1], j, m])
                n += 1
                pbar.update(n)

        results_name = raw_input('Give dataset name: ')
        Results(DB).save_matching_scores(results, results_name, training_data_name, target_data_name)

    # ####################################################
    # #   VIEW MATCH RESULTS
    # ####################################################
    def do_view_run_results(self, line):

        results_name = raw_input('Load matches: ')

        # #### LOAD MATCHING SCORES FROM DB
        results, trang_data_name, targt_data_name = Results(DB).load_matching_scores(results_name)

        # #### LOAD TARGET DATA AND TRAINING DATA
        targt_data = TargetData(DB).load(targt_data_name)
        trang_data = TrainingData(DB).load(trang_data_name)

        # #### VISUALIZE
        matching_review(results, targt_data, trang_data)

    # # ####################################################
    # # #   ANALYZE RESULTS
    # # ####################################################
    def do_analyze_run(self, line):

        results_name = raw_input('Load matching scores: ')

        # #### LOAD MATCHING SCORES FROM DB
        results, trang_data_name, targt_data_name = Results(DB).load_matching_scores(results_name)

        # #### LOAD TARGET DATA AND TRAINING DATA
        targt_data = TargetData(DB).load(targt_data_name)
        trang_data = TrainingData(DB).load(trang_data_name)

        # #### RUN ANALYSIS
        a = TrainingAnalysis(targt_data, targt_data_name, trang_data_name, results)
        a.plot(targt_data, trang_data)

    def do_EOF(self, line):
        return True

if __name__ == '__main__':
    DRconsole().cmdloop()
