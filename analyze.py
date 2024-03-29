import numpy as np
from math import log, sqrt, pi, gamma
from scipy.integrate import quad
from scipy.stats import skew as sk
from scipy.stats import linregress
from scipy.stats import norm
from scipy import pi, sqrt, exp
from scipy.special import erf
import pdb


def significance(data, pos_change, neg_change, confidence, max_error):
    """
    Coarse grains data into -1, 0, and 1 and determines whether there exists a statistically significant bias toward up (1), down (-1), or nothing (0)

    Inputs
    ------
    data -- 1d numpy array
    pos_change -- minimum value to be classified as "up" (1)
    neg_change -- minimum value to be classified as "down" (-1)
    confidence -- level of confidence (usually .95 or .99)
    max_error -- max error allowed from uniform distribution

    Outputs
    -------
    returns tuple:
    -- conclusive (True/False)
    -- direction (1, 0, -1)
    -- min. number of trials

    Notes
    ------
    Based on http://en.wikipedia.org/wiki/Checking_whether_a_coin_is_fair#Estimator_of_true_probability
    """

    sstring = []

    for value in data:
        if value <= neg_change:
            sstring.append(-1)
        if neg_change < value < pos_change:
            sstring.append(0)
        if value >= pos_change:
            sstring.append(1)

    sstring = np.array(sstring)

    n = len(sstring)
    r = 1./3.
    Z = norm.ppf(confidence)

    conclusive = False
    direction = None
    count = 0
    min_trials = None

    # Check -1

    # Check min n
    p = float((sstring == -1).sum())/float(n)
    min_n = (Z**2/max_error**2)*(p*(1-p))

    if (p > r + max_error):

        min_trials = min_n

        if (n >= min_n):
            conclusive = True
            direction = -1
            count += 1

    # Check 0

    # Check min n
    p = float((sstring == 0).sum())/float(n)
    min_n = (Z**2/max_error**2)*(p*(1-p))

    if (p > r + max_error):

        min_trials = min_n

        if (n >= min_n):
            conclusive = True
            direction = 0
            count += 1

    # Check 1

    # Check min n
    p = float((sstring == 1).sum())/float(n)
    min_n = (Z**2/max_error**2)*(p*(1-p))

    if (p > r + max_error):

        min_trials = min_n

        if (n >= min_n):
            conclusive = True
            direction = 1
            count += 1

    if count > 1:
        direction = None
        conclusive = False

    return conclusive, direction, min_trials


def autocorrelation(data, offset):
    """
    Estimate autocorrelation for discrete process with known mean and variance

    Inputs
    ------
    data -- 1d numpy array (time series)
    offset -- number of indeces to offset by

    Outputs
    -------
    returns estimate of autocorrelation

    Source
    ------
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation

    """
    mu = np.mean(data)
    var = np.var(data)

    return sum([(y-mu)*(yt-mu) for y, yt in zip(data, data[offset:])]) / ((len(data)-offset)*var)


def hurst(data, min_dset=8, return_error=False):
    """
    Estimate Hurst Exponent for a time series, a measure of long-term memory

    Inputs
    ------
    data -- 1d numpy array (time series)
    min_dset -- minimum time range for rescaled range calculation (default=8)
    return_error -- if set to true, returns standard error of estimate (default=False)

    Outputs
    -------
    returns estimate of hurst exponent and optionally the standard error

    Source
    ------
    http://www.bearcave.com/misl/misl_tech/wavelets/hurst/
    See also: http://en.wikipedia.org/wiki/Hurst_exponent

    """

    def rs(series):
        """Calculate rescaled range of data"""

        mean_adjusted_series = np.subtract(series, np.mean(series))
        cumulative_deviate_series = np.array([np.sum(mean_adjusted_series[:t]) \
                                              for t in range(len(mean_adjusted_series))])
        return np.ptp(cumulative_deviate_series) / np.std(series)

    def rs_vector(series):
        """
        Calculate vector x:

        x0 = region size
        x1 = RS ave.
        x2 = log(2)(region size)
        x3 = log(2)(RS ave.)
        """

        def recurse(sub):

            if len(sub) >= min_dset:

                rescaled_ranges.append([rs(sub), len(sub)])
                recurse(sub[:len(sub)/2])
                recurse(sub[len(sub)/2:])

        rescaled_ranges = []
        recurse(series)
        rescaled_ranges = np.array(rescaled_ranges)

        vector = np.unique(rescaled_ranges[:, 1])

        data = [np.mean(rescaled_ranges[np.where(rescaled_ranges[:, 1] == r)][:, 0]) for r in vector]
        vector = np.column_stack((vector, data))

        data = [log(region, 2) for region in vector[:, 0]]
        vector = np.column_stack((vector, data))

        data = [log(rsave, 2) for rsave in vector[:, 1]]
        vector = np.column_stack((vector, data))

        return vector

    vector = rs_vector(data)

    slope, intercept, r_value, p_value, std_err = linregress(vector[:, 2], vector[:, 3])

    if return_error:
        return slope, std_err
    else:
        return slope


def sharpe(returns, riskfree):
    """
    Sharpe ratio

    Inputs
    ------
    returns -- 1d numpy array with return values
    riskfree -- static riskfree return value

    Outputs
    -------
    returns sharpe ratio

    Source
    ------
    http://en.wikipedia.org/wiki/Sharpe_ratio

    """

    return (np.mean(returns)-riskfree) / np.std(returns)


def differences(data, method='log', normalized=False):
    """
    Calculate series of successive differences within time series

    Inputs
    ------
    data -- 1d numpy array
    method -- choose from log, pct, nominal

    Outputs
    -------
    returns 1d numpy array of differences

    """

    if method == 'log':
        output = [log(a2) - log(a1) for a2, a1 in zip(data[1:], data)]
    if method == 'pct':
        output = [(a2-a1)/a1 for a2, a1 in zip(data[1:], data)]
    if method == 'nominal':
        output = [a2-a1 for a2, a1, in zip(data[1:], data)]

    if normalized:
        mean = np.mean(output)
        std = np.std(output)
        output = [(rt - mean)/std for rt in output]

    return np.array(output)


def coarse_grain(data, method='zero_centered', bin_frac=.3):
    """
    Coarse-Grain data

    Inputs
    ------
    data -- 1d numpy array
    method -- choose from equal, zero_centered (default), mean_centered
    bin_frac -- percentage of y range to calculate boundaries for centered methods (default=.3)

    Outputs
    -------
    returns 1d numpy array containing symbol string

    Description
    -----------
    Turn data y1,y2,...,yn into a sequence i1, i2,...,iN of 1s, 2s, 3s, and 4s
    We use three kinds of coarse-graining:
    equal-size bins: divide the range of values into four intervals of equal length
    zero-centered bins: take 0 as the boundary between bins 2 and 3; place the other boundaries symmetrically above and below 0
    mean-centered bins: take the mean of the data to be the boundary between bins 2 and 3; place the other boundaries symmetrically above and below the mean
    median-centered bins: take the mean of the data to be the boundary between bins 2 and 3; place the other boundaries symmetrically above and below the mean

    Source
    ------
    http://classes.yale.edu/fractals/IntroToFrac/DrivenIFS/DataIFS/CoarseGrain/CoarseGrain.html

    """

    sstring = []
    mx = np.max(data)
    mn = np.min(data)

    if method == 'equal':
        for value in data:
            if (mn) <= value < (mn + .25*(mx-mn)):
                sstring.append(1)
            if (mn + .25*(mx-mn)) <= value < (mn + .5*(mx-mn)):
                sstring.append(2)
            if (mn + .5*(mx-mn)) <= value < (mn + .75*(mx-mn)):
                sstring.append(3)
            if (mn + .75*(mx-mn)) <= value <= (mx):
                sstring.append(4)
    if method == 'zero_centered':
        for value in data:
            if (mn) <= value < (-bin_frac*abs(mx-mn)):
                sstring.append(1)
            if (-bin_frac*abs(mx-mn)) <= value < (0):
                sstring.append(2)
            if (0) <= value < (bin_frac*abs(mx-mn)):
                sstring.append(3)
            if (bin_frac*abs(mx-mn)) <= value <= (mx):
                sstring.append(4)
    if method == 'mean_centered':
        cent = np.mean(data)
        for value in data:
            if (mn) <= value < (cent-bin_frac*abs(mx-mn)):
                sstring.append(1)
            if (cent-bin_frac*abs(mx-mn)) <= value < (cent):
                sstring.append(2)
            if (cent) <= value < (cent+bin_frac*abs(mx-mn)):
                sstring.append(3)
            if (cent+bin_frac*abs(mx-mn)) <= value <= (mx):
                sstring.append(4)
    if method == 'median_centered':
        cent = np.median(data)
        for value in data:
            if (mn) <= value < (cent-bin_frac*abs(mx-mn)):
                sstring.append(1)
            if (cent-bin_frac*abs(mx-mn)) <= value < (cent):
                sstring.append(2)
            if (cent) <= value < (cent+bin_frac*abs(mx-mn)):
                sstring.append(3)
            if (cent+bin_frac*abs(mx-mn)) <= value <= (mx):
                sstring.append(4)

    return np.array(sstring)


def driven_IFS(sstring):
    """
    Build an array for purposes of plotting a symbol string

    Inputs
    ------
    sequence -- 1d numpy array containing coarse-grained symbol string

    Outputs
    -------
    returns 2d numpy array with [x,y] plottable points

    """

    output = []
    current_point = [.5, .5]

    for symbol in sstring:
        if symbol == 1:
            current_point = [.5*current_point[0], .5*current_point[1]]
            output.append(current_point)
        if symbol == 2:
            current_point = [1-.5*current_point[0], .5*current_point[1]]
            output.append(current_point)
        if symbol == 3:
            current_point = [.5*current_point[0], 1-.5*current_point[1]]
            output.append(current_point)
        if symbol == 4:
            current_point = [1-.5*current_point[0], 1-.5*current_point[1]]
            output.append(current_point)

    return np.array(output)


def skew_normal_density(data):
    """
    Skew-normal probability density function over dataset

    Inputs
    ------
    data -- 1d numpy array

    Outputs
    -------
    returns tuple:
    output[0] -- 1d numpy array containing x linspace used to generate function
    output[1] -- 1d numpy array containing density function
    output[2] -- probability of positive change
    output[3] -- probability of negative change

    """

    def pdf(x):
        """Normal probability density function"""
        return 1/sqrt(2*pi) * exp(-x**2/2)

    def cdf(x):
        """Cumulative distribution function"""
        return (1+erf(x/sqrt(2))) / 2

    def skew(x, e=0, w=1, a=0):
        """Skew-normal probability density function"""
        t = (x-e) / w
        return 2 / w * pdf(t) * cdf(a*t)

    mu = np.mean(data)
    sigma = np.std(data)
    alpha = sk(data)
    x_plot = np.linspace(min(data), max(data), 1000)
    function = skew(x_plot, mu, sigma, alpha)
    ppos = quad(skew, 0, np.inf, args=(mu, sigma, alpha))
    pneg = quad(skew, -np.inf, 0, args=(mu, sigma, alpha))

    return x_plot, function, ppos, pneg


def q_gaussian_density(x, q, b):
    """q-gaussian probability sensity function
    http://en.wikipedia.org/wiki/Q-Gaussian_distribution"""

    def c(q):
        """normalization factor"""
        if -float('inf') < q < 1:
            return (2 * sqrt(pi) * gamma(1/(1-q))) / ((3 - q) * sqrt(1 - q) * gamma((3 - q) / (2 * (q - 1))))
        if q == 1:
            return sqrt(pi)
        if 1 < q < 3:
            return (sqrt(pi) * gamma((3-q)/2*(q-1))) / (sqrt(q-1) * gamma(1/(q-1)))

    def e(x, q):
        """q-exponential"""
        return (1 + (1 - q)*x) ** 1/(1-q)

    m = c(q)
    n = e(-b * x**2, q)
    return sqrt(b)/m * n


def return_range(data, start_index, end_index, method='log'):
    """
    Calculate return

    Inputs
    ------
    data -- 1d numpy array
    start_index -- index of start value
    end_index -- index of end value
    method -- choose from log, pct, nominal

    Outputs
    -------
    returns float of change value

    """

    start_val = float(data[start_index])
    end_val = float(data[end_index])

    if method == 'log':
        output = log(end_val) - log(start_val)
    if method == 'pct':
        output = (end_val - start_val) / start_val
    if method == 'nominal':
        output = end_val - start_val

    return output


def historical_volatility(data, diff_method='log'):
    """
    Calculate volatility

    Inputs
    ------
    data -- 1d numpy array
    method -- choose from log, pct, nominal for calculation of successive returns

    Outputs
    -------
    returns float of volatility

    """

    output = np.std(differences(data, method=diff_method))

    return output

