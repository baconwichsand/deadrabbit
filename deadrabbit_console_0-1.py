import sys
import cmd
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

DEFAULT_CFG = 'deadrabbit.cfg'

CFG = raw_input('Load config (leave empty for default: ' + DEFAULT_CFG + '): ')
if CFG == '':
    CFG = DEFAULT_CFG
if not os.path.isfile(CFG):
    LOADED = False
    while not LOADED:
        print '\'' + CFG + '\'' + ' not found'
        CFG = raw_input('Load config (.cfg file): ')
        if os.path.isfile(CFG):
            LOADED = True
print '====> config loaded from', CFG

DB = Config(CFG).get("General", 'hdf5')


class DRconsole(cmd.Cmd):

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
