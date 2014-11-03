import numpy as np
import pdb
import pylab as plt
from scipy.stats import skew, kurtosis, pearsonr
from analyze import sharpe, differences, significance, return_range, historical_volatility
from Visualization import plot_IFS, plot_difference_IFS, change_hist, range_hist, plot_density
from math import log
from matplotlib import rc


class TrainingAnalysis(object):
    """Analyze a set of samples on target data"""

    def __init__(self, targt_data, targt_data_name, trang_data_name, matches):

        def calculate():
            """Build data array"""

            # Expand for multiple timeframes
            data = np.repeat(matches, len(self._timefrm), axis=0)
            timeframe = np.array(self._timefrm*len(matches)).T

            # Add timeframes
            data = np.column_stack((data, timeframe))

            # Add change
            change = [return_range(targt_data, d, d+t) for d, t in zip(data[:, 1], data[:, 4])]
            data = np.column_stack((data, change))

            # Add subset volatility
            subsets = [targt_data[start:end] for start, end in zip(data[:, 0], data[:, 1])]
            volatility = [historical_volatility(s) for s in subsets]
            data = np.column_stack((data, volatility))

            # Add range
            rng = []
            for d, t in zip(data[:, 1], data[:, 4]):
                subset = targt_data[d:d+t]
                change = [log(r2/r1) for r2, r1 in zip(subset[1:], subset)]
                rng.append(np.std(change))

            data = np.column_stack((data, rng))

            # Add match length
            length = [d[1]-d[0] for d in data]
            data = np.column_stack((data, length))

            return data

        # Timeframes (NOT A RANGE, INDIVIDUAL VALUES!)
        self._timefrm = [30, 50, 60, 70, 80, 85, 90, 100]

        # Match score filter
        self._scorefilter = 0.8

        # Dataset identifiers
        self.trang_data_name = trang_data_name
        self.targt_data_name = targt_data_name

        # Data array
        # data[:,0] = target data subset start index
        # data[:,1] = target data subset end index
        # data[:,2] = training data id
        # data[:,3] = matching score
        # data[:,4] = time frame
        # data[:,5] = change
        # data[:,6] = target data subset volatility
        # data[:,7] = range
        # data[:,8] = match length
        self.data = calculate()

        # List of sample ids
        self.samples = np.unique(self.data[:, 2])

        # Index of current sample
        self.index = 0

    def draw(self, targt_data, trang_data):
        """Plot analysis of invididual samples"""

        # Current sample
        sample = self.data[np.where(self.data[:, 2] == self.samples[self.index])]

        # Filter above specific matching score
        sample = sample[np.where(sample[:, 3] > self._scorefilter)]

        def summary(plot):
            """Draws summary of current sample analysis"""

            text_string = []

            # Market
            text_string.append(self.targt_data_name)

            # Pattern class
            text_string.append(self.trang_data_name)

            # Pattern id
            text_string.append(str(int(self.samples[self.index])))

            # Score filter
            text_string.append(str(self._scorefilter))

            # Avg. score
            text_string.append(str(round(np.mean(sample[:, 3]), 2)))

            # Number of matches
            text_string.append(str(len(sample)/len(self._timefrm)))

            # Min match length
            text_string.append(str(np.min(sample[:, 8])))

            # Max match length
            text_string.append(str(np.max(sample[:, 8])))

            # Avg. match length
            text_string.append(str(round(np.mean(sample[:, 8]), 2)))

            # Significance
            cdata = sample[np.where(sample[:, 4] == self._timefrm[-1])][:, 5]
            pos_change = .05
            neg_change = -.05
            confidence = .95
            max_error = 0.065
            sig, direction, min_trials = significance(cdata, pos_change, neg_change, confidence, max_error)
            text_string.append(str(pos_change))
            text_string.append(str(neg_change))
            text_string.append(str(confidence))
            text_string.append(str(max_error))
            text_string.append(str(int(min_trials)))
            text_string.append(str(sig))
            if direction is None:
                text_string.append('N/A')
            else:
                text_string.append(str(direction))

            plot.text(0, 0, r'\begin{tabular}{lc} {Market: } & {%s} \\ {Pattern class: } & {%s} \\ {Pattern id: } & {%s} \\ \\ {Score filter: } & {%s} \\ {Avg. score: } & {%s} \\ {Number of matches: } & {%s} \\ \\ {Min. match length: } & {%s} \\ {Max. match length: } & {%s} \\ {Avg. match length: } & {%s} \\ \\ {Pos. threshold: } & {%s} \\ {Neg. threshold: } & {%s} \\ {Confidence: } & {%s} \\ {Max error: } & {%s} \\ {Min. trials: } & {%s} \\ {Significant?: } & {%s} \\ {Direction: } & {%s} \\ \end{tabular}' % tuple(text_string), fontsize=12)

            plot.axis('off')

        def plot_stats(plot):

            text_string = []

            # Timeframes
            for tmfrm in self._timefrm:
                text_string.append(str(tmfrm))

            # Change max
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = np.max(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            # Change min
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = np.min(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            # Change mean
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = np.mean(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            # Change median
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = np.median(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            # Change std
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = np.std(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            # Sharpe
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = sharpe(cdata[:, 5], 0)
                text_string.append(str(round(val, 4)))

            # Change skew
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = skew(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            # Change kurtosis
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = kurtosis(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            # Prob + change
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = float(len(cdata[np.where(cdata[:, 5] > 0)])) / float(len(cdata[:, 5]))
                text_string.append(str(round(val, 4)))

            # Mean + change
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = np.mean(cdata[np.where(cdata[:, 5] > 0)][:, 5])
                text_string.append(str(round(val, 4)))

            # Prob - change
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = float(len(cdata[np.where(cdata[:, 5] < 0)])) / float(len(cdata[:, 5]))
                text_string.append(str(round(val, 4)))

            # Prob - change
            for tmfrm in self._timefrm:
                cdata = sample[np.where(sample[:, 4] == tmfrm)]
                val = np.mean(cdata[np.where(cdata[:, 5] < 0)][:, 5]) / len(cdata[:, 5])
                text_string.append(str(round(val, 4)))

            plot.text(.0, 0.1, r'\def\arraystretch{1.2} \begin{tabular}{|l|c|c|c|c|c|c|c|c|} \hline {timeframe} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {chg. max} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {chg. min} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {chg. mean} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {chg. median} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {chg. std.d.} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {sharpe} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {chg. skew} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {chg. kurt.} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {prob. +} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {avg. + chg.} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s}\\ \hline {prob. -} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline {avg. - chg.} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} & {%s} \\ \hline \end{tabular}' % tuple(text_string), fontsize=11)

            plot.axis('off')

        def results(plot):
            """Plot graph of results"""

            # Nominal changes
            data = sample[np.where(sample[:, 4] == self._timefrm[0])]
            nomres = [np.insert(targt_data[y-1:y+np.max(self._timefrm)],\
                                   0, targt_data[y-1]) for y in data[:, 1]]

            # Percentage changes
            pctres = np.array([np.log(np.divide(y[1:], y[0])) for y in nomres])

            # Plot percentage changes
            plot.hold(True)
            plot.axis('off')
            for res in pctres:
                plot.plot(res, alpha=.15)

            # Plot average change
            avgres = [np.mean(r) for r in pctres.T]
            plot.plot(avgres, 'k--', lw=2)

            # Plot zero line
            plot.plot(np.zeros(len(avgres)), 'r--', lw=2)
        
            # Plot timeframe lines
            plot.plot([0, 0], [np.min(pctres), np.max(pctres)], 'k:')
            plot.annotate(r'$t_0 = %s$' % str(0), xy=(0, np.max(pctres)+0.03), horizontalalignment='center')
            for i, tframe in enumerate(self._timefrm):

                plot.plot([tframe, tframe], [np.min(pctres), np.max(pctres)], 'k:')
                plot.annotate(r'$t_%s = %s$' % (i+1, tframe), xy=(tframe, np.max(pctres)+0.03), horizontalalignment='center')

        ##### PLOT PATTERN
        self.pattern.cla()
        self.pattern.axis('off')
        self.pattern.plot(trang_data[self.index], 'k')

        ##### PLOT GRAPH OF RESULTS
        self.resplot.cla()
        results(self.resplot)

        ##### PLOT CHANGE HISTOGRAMS
        for i, hist in enumerate(self.histograms):
            hist.cla()
            data = sample[np.where(sample[:, 4] == self._timefrm[i])][:, 5]
            change_hist(hist, data)

        ##### PLOT CHANGE RANGE HISTOGRAM
        for i, prob in enumerate(self.probplots):
            prob.cla()
            data = sample[np.where(sample[:, 4] == self._timefrm[i])][:, 7]
            range_hist(prob, data)

        ##### PLOT SUMMARY
        self.summplot.cla()
        summary(self.summplot)

        ##### PLOT STATS
        self.stats.cla()
        plot_stats(self.stats)

        ##### PLOT PROB. DENSITY FUNCTION
        self.scat3.cla()

        ##### PLOT SCORE / CHANGE SCATTER
        self.scat4.cla()
        self.scat4.set_xlabel('match score')
        self.scat4.set_ylabel('change')
        self.scat4.scatter(sample[:, 3], sample[:, 5], marker='.', s=0.5)
        self.scat4.plot([self.scat4.get_xlim()[0], self.scat4.get_xlim()[1]], [0, 0], 'r--')

        ##### PLOT MATCH LENGTH / CHANGE SCATTER
        self.scat5.cla()
        self.scat5.set_xlabel('match timeframe')
        self.scat5.set_ylabel('change')
        self.scat5.scatter(sample[:, 8], sample[:, 5], marker='.', s=0.5)
        self.scat5.plot([self.scat5.get_xlim()[0], self.scat5.get_xlim()[1]], [0, 0], 'r--')

        plt.draw()

    def on_key(self, event, targt_data, trang_data):
        """Cycle through individual samples with arrow keys"""

        if event.key == 'right' and self.index+1 < len(self.samples):
            self.index += 1
            self.draw(targt_data, trang_data)

        if event.key == 'left' and self.index-1 >= 0:
            self.index -= 1
            self.draw(targt_data, trang_data)

        if event.key == 'escape':
            plt.close()

    def plot(self, targt_data, trang_data):
        """Base plot function, build figure and subplots"""

        # Set LaTeX interpreter and font
        rc('font', family='serif')
        rc('text', usetex=True)

        self.fig = plt.figure()

        # Fill entire window
        self.fig.subplots_adjust(left=0.03, bottom=0.05, right=.97, top=.95, wspace=.55, hspace=.2)

        # Enable scrolling through samples with arrow keys
        self.fig.canvas.mpl_connect('key_press_event', lambda event: \
                                    self.on_key(event, targt_data, trang_data))

        # Pattern
        self.pattern = plt.subplot2grid((6, 10), (0, 0), colspan=2, rowspan=2)

        # Build subplots
        self.resplot = plt.subplot2grid((6, 10), (0, 2), colspan=8, rowspan=2)


        # Histograms
        self.histograms = [plt.subplot2grid((6, 10), (2, 2)), \
                           plt.subplot2grid((6, 10), (2, 3)), \
                           plt.subplot2grid((6, 10), (2, 4)), \
                           plt.subplot2grid((6, 10), (2, 5)), \
                           plt.subplot2grid((6, 10), (2, 6)), \
                           plt.subplot2grid((6, 10), (2, 7)), \
                           plt.subplot2grid((6, 10), (2, 8)), \
                           plt.subplot2grid((6, 10), (2, 9))]

        # Probability density plots
        self.probplots = [plt.subplot2grid((6, 10), (3, 2)), \
                          plt.subplot2grid((6, 10), (3, 3)), \
                          plt.subplot2grid((6, 10), (3, 4)), \
                          plt.subplot2grid((6, 10), (3, 5)), \
                          plt.subplot2grid((6, 10), (3, 6)), \
                          plt.subplot2grid((6, 10), (3, 7)), \
                          plt.subplot2grid((6, 10), (3, 8)), \
                          plt.subplot2grid((6, 10), (3, 9))]

        # Summary
        self.summplot = plt.subplot2grid((6, 10), (2, 0), colspan=2, rowspan=2)

        # Stats
        self.stats = plt.subplot2grid((6, 10), (4, 0), colspan=3, rowspan=2)

        # Scatterplots
        self.scat3 = plt.subplot2grid((6, 10), (4, 4), colspan=2, rowspan=2)
        self.scat4 = plt.subplot2grid((6, 10), (4, 6), colspan=2, rowspan=2)
        self.scat5 = plt.subplot2grid((6, 10), (4, 8), colspan=2, rowspan=2)

        self.draw(targt_data, trang_data)
        plt.show()


            # ###### SINGLE METRIC STATS
            # xpos = 0.32
            # ypos = 0.32

            # information.text(xpos, ypos, r'\begin{tabular}{|l|c|c|c|} \hline \multicolumn{4}{|c|}{\textbf{Single Metric Stats}} \\ \hline {x} & {Change} & {Vol} & {Range} \\ \hline min_x &' + str(round(self.stats['change']['min'], 2)) + r'&' + str(round(self.stats['volatility']['min'], 2)) + r'&' + str(round(self.stats['range']['min'], 2)) + r'\\ \hline max_x &' + str(round(self.stats['change']['max'], 2)) + r'&' + str(round(self.stats['volatility']['max'], 2)) + r'&' + str(round(self.stats['range']['max'], 2)) + r'\\ \hline \overline{x} &' + str(round(self.stats['change']['mean'], 2)) + r'&' + str(round(self.stats['volatility']['mean'], 2)) + r'&' + str(round(self.stats['range']['mean'], 2)) + r'\\ \hline \widetilde{x} &' + str(round(self.stats['change']['median'], 2)) + r'&' + str(round(self.stats['volatility']['median'], 2)) + r'&' + str(round(self.stats['range']['median'], 2)) + r'\\ \hline \sigma_x &' + str(round(self.stats['change']['std'], 2)) + r'&' + str(round(self.stats['volatility']['std'], 2)) + r'&' + str(round(self.stats['range']['std'], 2)) + r'\\ \hline \gamma_x &' + str(round(self.stats['change']['skew'], 2)) + r'&' + str(round(self.stats['volatility']['skew'], 2)) + r'&' + str(round(self.stats['range']['skew'], 2)) + r'\\ \hline \kappa_x &' + str(round(self.stats['change']['kurtosis'], 2)) + r'&' + str(round(self.stats['volatility']['kurtosis'], 2)) + r'&' + str(round(self.stats['range']['kurtosis'], 2)) + r'\\ \hline \end{tabular}')


# #### TIME DELAY (MIN, MAX, STEP)
# DELAY = [0, 200, 5]

# #### TIME DURATION (MIN, MAX, STEP)
# DURATION = [20, 200, 5]

# class IndividualMatch(object):
#     """Represents analysis of match between training and target subset"""

#     def __init__(self, target_data, target_data_name, training_data_id, \
#                  subset_index_start, subset_index_end, score, transformations):

#         #### TIME PARAMETERS
#         self.__path_timeframes = np.append(DELAY, DURATION)
#         self.__indeces = [subset_index_start, subset_index_end, \
#                           subset_index_end + self.__path_timeframes[1] + \
#                           self.__path_timeframes[4]]

#         #### DATASET IDENTIFIERS
#         # self.target_data_id = target_data_name
#         # self.training_data_id = training_data_id

#         self.__target_data = target_data
#         self.target_data_id = target_data_name
#         self.training_data_id = training_data_id

#         #### MATCHING VARIABLES
#         self.score = score
#         self.transformations = transformations

#         START_COMPUTE = time.time()
#         self.__data = self.build_data()
#         self.__stats = {'delay': basic_stats(self.__data[:, 2]),
#                         'duration': basic_stats(self.__data[:, 3]),
#                         'change': basic_stats(self.__data[:, 4]),
#                         'volatility': basic_stats(self.__data[:, 5]),
#                         'range': basic_stats(self.__data[:, 6])}
#         self.__compute_time = time.time()-START_COMPUTE

#     def build_data(self):
#         """Build paths"""
#         return np.array([[self.__indeces[1]+delay, \
#                           self.__indeces[1]+delay+duration, \
#                           delay, duration, \
#                           self.__target_data[self.__indeces[1]:\
#                                              self.__indeces[1]+delay+duration][-1]/\
#                           self.__target_data[self.__indeces[1]+delay:\
#                                              self.__indeces[1]+delay+duration][0]-1, \
#                           np.std([r1/r2-1 \
#                                   for r1, r2 in zip(self.__target_data\
#                                                     [self.__indeces[1]+delay:\
#                                                      self.__indeces[1]+delay+duration][1:], \
#                                                     self.__target_data\
#                                                     [self.__indeces[1]+delay:\
#                                                      self.__indeces[1]+delay+duration])]), \
#                           np.ptp(self.__target_data\
#                                  [self.__indeces[1]+delay:\
#                                   self.__indeces[1]+delay+duration])]\
#                          for delay in range(self.__path_timeframes[0], \
#                                             self.__path_timeframes[1], \
#                                             self.__path_timeframes[2]) \
#                          for duration in range(self.__path_timeframes[3], \
#                                                self.__path_timeframes[4], \
#                                                self.__path_timeframes[5])])


#     def visualize(self):
#         """Visualize analysis"""

#         def paths_simple(plot):
#             """Draw paths"""

#             # PLOT TARGET DATA
#             plot.plot(self.__target_data[self.__indeces[0]:\
#                                          self.__indeces[2]], color='black')

#             # VERTICAL LINE TO INDICATE END OF SUBSET
#             plot.plot([self.__indeces[1], self.__indeces[1]], \
#                       [min(self.__target_data[self.__indeces[0]:\
#                                               self.__indeces[2]]), \
#                        max(self.__target_data[self.__indeces[0]:\
#                                               self.__indeces[2]])], 'b', lw=1)

#             # PLOT PATHS
#             for i in range(self.paths):

#                 # CHECK IF CHANGE IS POSITIVE FOR DASH COLOR
#                 if self.change[i] >= 0:
#                     color = 'g'
#                 else:
#                     color = 'r'
#                 plot.plot([self.start_index[i], self.end_index[i]], \
#                           [self.__target_data[self.start_index[i]], \
#                            self.__target_data[self.end_index[i]]], color + '--')


#         def histogram(plot, metric):
#             """Draw histogram"""

#             x = getattr(self, metric)
#             mu = self.__stats[metric]['mean']
#             sigma = self.__stats[metric]['std']
#             x_plot = np.linspace(min(x), max(x), 1000)
#             pdf = stats.norm.pdf(x_plot, mu, sigma)


#             plot.hist(x, bins=50, normed=True, color='black', alpha=0.3, histtype='stepfilled', label='data')
#             plot.plot(x_plot, pdf, 'b--', label='pdf', lw=2)
#             plot.legend(loc='best')
#             plot.grid(True)
#             plot.set_title(metric)

#             if metric == 'change':
#                 plot.fill_between(x_plot, pdf, where=x_plot<0, interpolate=True, color='red', alpha=0.5)
#                 plot.fill_between(x_plot, pdf, where=x_plot>0, interpolate=True, color='green', alpha=0.5)


#         rc('text', usetex=True)
#         fig = plt.figure()

#         ##### PLOT PATHS
#         p_simple = fig.add_subplot(321)
#         p_simple.grid(True)
#         p_simple.set_title('paths')
#         paths_simple(p_simple)

#         # ##### PLOT INFORMATION
#         information = fig.add_subplot(322)
#         information.axis('off')

#         ###### GENERAL TABLE
#         xpos = 0.02
#         ypos = 0.53
#         information.text(xpos, ypos, r'\begin{tabular}{|l|c|} \hline \multicolumn{2}{|c|}{\textbf{General}} \\ \hline {Target Data ID} &' +str(self.target_data_id) + r'\\ \hline {Training Data ID} &' + str(self.training_data_id) + r'\\ \hline Match Score &' + str(round(self.score, 2)) + r'\\ \hline Paths &' + str(self.paths) + r'\\ \hline {Comp. Speed (ms)} &' + str(round(self.__compute_time*1000, 2)) + r'\\ \hline \end{tabular}')

#         ###### SINGLE METRIC STATS
#         xpos = 0.32
#         ypos = 0.32

#         information.text(xpos, ypos, r'\begin{tabular}{|l|c|c|c|c|c|} \hline \multicolumn{6}{|c|}{\textbf{Single Metric Stats}} \\ \hline {x} & {Delay} & {Duration} & {Change} & {Vol} & {Range} \\ \hline min_x &' + str(int(self.__stats['delay']['min'])) + r'&' + str(int(self.__stats['duration']['min'])) + r'&' + str(round(self.__stats['change']['min'], 2)) + r'&' + str(round(self.__stats['volatility']['min'], 2)) + r'&' + str(round(self.__stats['range']['min'], 2)) + r'\\ \hline max_x &' + str(int(self.__stats['delay']['max'])) + r'&' + str(int(self.__stats['duration']['max'])) + r'&' + str(round(self.__stats['change']['max'], 2)) + r'&' + str(round(self.__stats['volatility']['max'], 2)) + r'&' + str(round(self.__stats['range']['max'], 2)) + r'\\ \hline \overline{x} &' + str(round(self.__stats['delay']['mean'], 2)) + r'&' + str(round(self.__stats['duration']['mean'], 2)) + r'&' + str(round(self.__stats['change']['mean'], 2)) + r'&' + str(round(self.__stats['volatility']['mean'], 2)) + r'&' + str(round(self.__stats['range']['mean'], 2)) + r'\\ \hline \widetilde{x} & {-} & {-} &' + str(round(self.__stats['change']['median'], 2)) + r'&' + str(round(self.__stats['volatility']['median'], 2)) + r'&' + str(round(self.__stats['range']['median'], 2)) + r'\\ \hline \sigma_x & {-} & {-} &' + str(round(self.__stats['change']['std'], 2)) + r'&' + str(round(self.__stats['volatility']['std'], 2)) + r'&' + str(round(self.__stats['range']['std'], 2)) + r'\\ \hline \gamma_x & {-} & {-} &' + str(round(self.__stats['change']['skew'], 2)) + r'&' + str(round(self.__stats['volatility']['skew'], 2)) + r'&' + str(round(self.__stats['range']['skew'], 2)) + r'\\ \hline \kappa_x & {-} & {-} &' + str(round(self.__stats['change']['kurtosis'], 2)) + r'&' + str(round(self.__stats['volatility']['kurtosis'], 2)) + r'&' + str(round(self.__stats['range']['kurtosis'], 2)) + r'\\ \hline \end{tabular}')

#         ##### SINGLE

#         ##### PLOT HISTOGRAMS
#         # Change
#         change_hist = fig.add_subplot(323)
#         histogram(change_hist, 'change')

#         # Volatility
#         change_hist = fig.add_subplot(324)
#         histogram(change_hist, 'volatility')

#         ##### PLOT SCATTER
#         # Change v. Delay
#         plt.show()

#     @property
#     def paths(self):
#         """Returns the number of paths"""
#         return self.__data.shape[0]

#     @property
#     def start_index(self):
#         """Returns start index array"""
#         return self.__data[:, 0].astype(int)

#     @property
#     def end_index(self):
#         """Returns end index array"""
#         return self.__data[:, 1].astype(int)

#     @property
#     def delay(self):
#         """Returns delay array"""
#         return self.__data[:, 2].astype(int)

#     @property
#     def duration(self):
#         """Returns duration array"""
#         return self.__data[:, 3].astype(int)

#     @property
#     def change(self):
#         """Returns change array"""
#         return self.__data[:, 4].astype(float)

#     @property
#     def volatility(self):
#         """Returns volatility array"""
#         return self.__data[:, 5].astype(float)

#     @property
#     def range(self):
#         """Returns range array"""
#         return self.__data[:, 6].astype(float)
