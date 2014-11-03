import pdb
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import skew, kurtosis, probplot, gaussian_kde
from matplotlib import rc
from DataManipulation import normalize, interpolate
from RandGen import rand_range, generate, brownian
from Visualization import plot_difference_IFS, change_hist, plot_density, plot_IFS, lagplot, return_period_memory, autocorrelation_plot
from analyze import differences, coarse_grain, driven_IFS, skew_normal_density, hurst
import numpy as np
import pylab as plt


def simple_brownian(num_points):
    """Wiener process"""

    global generator
    generator = 'Brownian (Wiener)'

    return brownian(100, num_points, 1, 1)


def multifractal(num_points, dt1, holder):
    """Multifractal cartoon"""

    global generator, params
    generator = 'Multifractal'
    params = dt1, holder

    output = interpolate(generate(dt1, holder, 0.0001))
    return output[:num_points]


def numpy_random(num_points):
    """Generates random points between 0 and 1"""

    global generator
    generator = 'Numpy Random'

    return np.random.random(num_points)


def straight_line(num_points, increment):
    """Straight line with increment"""

    global generator, params
    generator = 'Straight line'
    params = str(increment)

    output = np.empty(num_points)
    output.fill(1000.)
    output = np.array([output[0] + increment*n for n in range(len(output))])

    return output


def rand_rng(num_points, minint, maxint):
    """Rand range generator"""

    global generator, params
    generator = 'Rand range'
    params = (minint, maxint)

    output = rand_range(num_points, minint, maxint)
    return output


fig = plt.figure()
fig.subplots_adjust(left=.02, bottom=.02, right=.98, top=.98, wspace=.35, hspace=.35)
rc('font', family='serif')
rc('text', usetex=True)


# Target Data
plt1 = plt.subplot2grid((6, 8), (0, 0), colspan=8, rowspan=1)

# Target Data info
plt2 = plt.subplot2grid((6, 8), (1, 0), colspan=1, rowspan=2)

# Target Data stuff 
plt3 = plt.subplot2grid((6, 8), (1, 1), colspan=1, rowspan=1)
plt4 = plt.subplot2grid((6, 8), (1, 2), colspan=1, rowspan=1)
plt5 = plt.subplot2grid((6, 8), (1, 3), colspan=1, rowspan=1)
plt6 = plt.subplot2grid((6, 8), (1, 4), colspan=1, rowspan=1)
plt7 = plt.subplot2grid((6, 8), (1, 5), colspan=1, rowspan=1)
plt8 = plt.subplot2grid((6, 8), (1, 6), colspan=1, rowspan=1)
plt9 = plt.subplot2grid((6, 8), (1, 7), colspan=1, rowspan=1)
plt10 = plt.subplot2grid((6, 8), (2, 1), colspan=1, rowspan=1)
plt11 = plt.subplot2grid((6, 8), (2, 2), colspan=1, rowspan=1)
plt12 = plt.subplot2grid((6, 8), (2, 3), colspan=1, rowspan=1)
plt13 = plt.subplot2grid((6, 8), (2, 4), colspan=1, rowspan=1)
plt14 = plt.subplot2grid((6, 8), (2, 5), colspan=1, rowspan=1)
plt15 = plt.subplot2grid((6, 8), (2, 6), colspan=1, rowspan=1)
plt16 = plt.subplot2grid((6, 8), (2, 7), colspan=1, rowspan=1)

# Differences
plt17 = plt.subplot2grid((6, 8), (3, 0), colspan=8, rowspan=1)

# Differences info
plt18 = plt.subplot2grid((6, 8), (4, 0), colspan=1, rowspan=2)

# Differences stuff
plt19 = plt.subplot2grid((6, 8), (4, 1), colspan=1, rowspan=1)
plt20 = plt.subplot2grid((6, 8), (4, 2), colspan=1, rowspan=1)
plt21 = plt.subplot2grid((6, 8), (4, 3), colspan=1, rowspan=1)
plt22 = plt.subplot2grid((6, 8), (4, 4), colspan=1, rowspan=1)
plt23 = plt.subplot2grid((6, 8), (4, 5), colspan=1, rowspan=1)
plt24 = plt.subplot2grid((6, 8), (4, 6), colspan=1, rowspan=1)
plt25 = plt.subplot2grid((6, 8), (4, 7), colspan=1, rowspan=1)
plt26 = plt.subplot2grid((6, 8), (5, 1), colspan=1, rowspan=1)
plt27 = plt.subplot2grid((6, 8), (5, 2), colspan=1, rowspan=1)
plt28 = plt.subplot2grid((6, 8), (5, 3), colspan=1, rowspan=1)
plt29 = plt.subplot2grid((6, 8), (5, 4), colspan=1, rowspan=1)
plt30 = plt.subplot2grid((6, 8), (5, 5), colspan=1, rowspan=1)
plt31 = plt.subplot2grid((6, 8), (5, 6), colspan=1, rowspan=1)
plt32 = plt.subplot2grid((6, 8), (5, 7), colspan=1, rowspan=1)

######### PLOT

num_points = 5000
generator = ''
params = ()

# Plot target data
# rdata = normalize(rand_rng(num_points, -100, 100))
# rdata = normalize(simple_brownian(num_points))
rdata = normalize(multifractal(num_points, 0.3, 0.3))
plt1.plot(rdata, 'k')

# Target Data info
plt2.axis('off')
plt2.text(.03, .35, r'\begin{tabular}{lc} \\ {gen.: %s} & \\ {params: %s} \\ {points: %s} & \\ \\ {min: %s} \\ {max: %s} \\ {mean: %s} & \\ {median: %s} & \\ {std. dev.: %s} & \\ {skew: %s} & \\ {kurtosis: %s} \\ \\ {hurst: %s} \end{tabular}' % (generator, params, str(num_points), str(round(np.min(rdata), 4)), str(round(np.max(rdata), 4)), str(round(np.mean(rdata), 4)), str(round(np.median(rdata), 4)), str(round(np.std(rdata), 4)), str(round(skew(rdata), 4)), str(round(kurtosis(rdata), 4)), str(round(hurst(rdata), 4))))

# Histogram
plt3.hist(rdata, bins=50)
plt3.set_title('value histogram')

# Lag plot
lag = 1
lagplot(plt4, rdata, lag)

# Autocorrelation
max_lag = 100
autocorrelation_plot(plt5, rdata, max_lag)

# Return periods vs. holder
max_timeframe = 30
return_period_memory(plt6, rdata, max_timeframe)

# Plot differences
diff = differences(rdata, method='nominal')
plt17.plot(diff, 'k')

# Diff info
plt18.axis('off')
plt18.text(.03, .53, r'\begin{tabular}{lc} \\ {min: %s} \\ {max: %s} \\ {mean: %s} & \\ {median: %s} & \\ {std. dev.: %s} & \\ {skew: %s} & \\ {kurtosis: %s} \\ \\ {hurst: %s} \end{tabular}' % (str(round(np.min(diff), 4)), str(round(np.max(diff), 4)), str(round(np.mean(diff), 4)), str(round(np.median(diff), 4)), str(round(np.std(diff), 4)), str(round(skew(diff), 4)), str(round(kurtosis(diff), 4)), str(round(hurst(diff), 4))))

# Histogram
plt19.hist(diff, bins=50)
hi, bin_edges = np.histogram(diff, bins=50)
plt19.set_title('diff histogram')

# Lagplot
lag = 1
lagplot(plt21, diff, lag)

# Autocorrelation
max_lag = 100
autocorrelation_plot(plt22, diff, max_lag)

# Probability plot
distribution = 'norm'
probplot(diff, dist=distribution, plot=plt23)
plt23.set_title('Gaussian?')

# Driven IFS
method = 'zero_centered'
bin_frac = .3
dt = driven_IFS(coarse_grain(diff, method=method, bin_frac=bin_frac))
plot_IFS(plt24, dt)
plt24.set_title('%s, %s' % (method, str(bin_frac)))

plt.show()
