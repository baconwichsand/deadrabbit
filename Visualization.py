import pylab as plt
import numpy as np
from analyze import hurst, autocorrelation
from FeatIdent import FeatureIdentifier
from DataManipulation import normalize, interpolate, eq_scale
from matplotlib.widgets import Slider, Button, RadioButtons


class Namespace:
    pass


def matching_review(results, targt_data, trang_data):

    print ''
    print 'Welcome to Matching Review! Available keys:'
    print ''
    print ' - Arrow keys (right, left) to step through matches'
    print ' - \'esc\' to exit matching review'

    fig, plt1 = plt.subplots(1, 1)

    results = results[results[:, 3].argsort()][::-1]

    ns = Namespace()
    ns.index = 0

    def plot_wnd():
        plt1.cla()
        plt1.grid(True)
        plt1.hold(True)
        trdata = trang_data[int(results[ns.index][2])]
        tadata = targt_data[int(results[ns.index][0]):int(results[ns.index][1])]
        tadata = normalize(tadata)
        output = eq_scale(trdata, tadata)
        plt1.plot(output[0])
        plt1.plot(output[1])
        plt.draw()
        print results[ns.index][3]

    def on_key(event):
        if event.key == 'right' and ns.index < len(results):
            ns.index += 1
            plot_wnd()

        if event.key == 'left' and ns.index > 0:
            ns.index -= 1
            plot_wnd()

        if event.key == 'escape':
            plt.close()
            print '====> End matching review'

    fig.canvas.mpl_connect('key_press_event', on_key)
    plot_wnd()
    plt.draw()
    plt.show()


def feature_selection(params, td):

    print ''
    print 'Welcome to feature selection!'
    print ''

    output = []

    l = params[0]
    s = params[1]
    zigzag = params[2]
    fractal = params[3]

    fig, (plt1, plt2) = plt.subplots(2, 1)

    ##### PLOT GRAPHS

    plt1.hold(True)
    plt1.grid(True)
    plt2.grid(True)
    plt1.plot(td)
    subline_l, = plt1.plot([s, s], [min(td), max(td)], 'r', lw=1)
    subline_r, = plt1.plot([s+l, s+l], [min(td), max(td)], 'r', lw=1)
    tsub = normalize(td[s:s+l])
    ssub = normalize(interpolate(FeatureIdentifier(td[s:s+l]).\
                                      simplify(fractal)))
    tp, = plt2.plot(np.arange(s, s+l), tsub, color='b')
    sp, = plt2.plot(np.arange(s, s+l), ssub, color='r')

    ##### BUILD SLIDERS

    axcol = 'lightgoldenrodyellow'
    axl = plt.axes([0.125, 0.48, 0.775, 0.015], axisbg=axcol)
    axs = plt.axes([0.125, 0.50, 0.775, 0.015], axisbg=axcol)
    axss = plt.axes([0.125, 0.04, 0.5, 0.02], axisbg=axcol)
    sld_l = Slider(axl, 'Subset Length', 5, 1000, valinit=l, valfmt='%0.0f')
    sld_s = Slider(axs, 'Subset Start', 0, len(td), \
                  valinit=s, valfmt='%0.0f')
    sld_ss = Slider(axss, 'Simpl. Error', 0.00, 0.5, valinit=zigzag)

    def update(val):
        """Update sliders"""
        if sld_s.val+sld_l.val < len(td):
            subline_l.set_xdata([int(sld_s.val), int(sld_s.val)])
            subline_r.set_xdata([int(sld_s.val+sld_l.val), \
                                 int(sld_s.val+sld_l.val)])
            plt2.set_xlim(int(sld_s.val), int(sld_s.val+sld_l.val))
            tp.set_xdata(np.arange(int(sld_s.val), int(sld_s.val+sld_l.val)))
            tp.set_ydata(normalize(td[int(sld_s.val):\
                                                 int(sld_s.val+sld_l.val)]))
            sp.set_xdata(np.arange(int(sld_s.val), int(sld_s.val+sld_l.val)))
            sp.set_ydata(normalize(interpolate(FeatureIdentifier(td[int(sld_s.val):int(sld_s.val+sld_l.val)]).simplify(sld_ss.val))))
            plt.draw()

    sld_s.on_changed(update)
    sld_l.on_changed(update)
    sld_ss.on_changed(update)

    ##### BUILD RESET BUTTON

    resetax = plt.axes([0.67, 0.03, 0.065, 0.035])
    resbutton = Button(resetax, 'Reset Graph', color=axcol, hovercolor='0.0975')

    def reset(event):
        sld_s.reset()
        sld_l.reset()
        sld_ss.reset()

    resbutton.on_clicked(reset)

    ##### BUILD SUBMIT BUTTON

    submitax = plt.axes([0.75, 0.03, 0.065, 0.035])
    subbutton = Button(submitax, 'Submit Data', color=axcol, hovercolor='0.0975')
    def submit(event):
        output.append(interpolate(FeatureIdentifier(td[sld_s.val:sld_s.val+sld_l.val]).simplify(sld_ss.val)))
        print "====> TrainingData added"
    subbutton.on_clicked(submit)

    ##### BUILD EXIT BUTTON

    exitax = plt.axes([0.83, 0.03, 0.065, 0.035])
    exitbutton = Button(exitax, 'Exit', color=axcol, hovercolor='0.0975')
    def leave(event):
        plt.close()
    exitbutton.on_clicked(leave)

    plt.draw()
    plt.show()

    return np.array(output)


def plot_IFS(plot, data):
    """Plot driven IFS analysis"""
    """"""
    """Inputs:"""
    """plot -- matplotlib plot"""
    """data -- 2d numpy array of [x,y] points returned by analyze.driven_IFS"""

    plot.scatter(data[:, 0], data[:, 1], marker='.', s=0.5)

    # plot.plot([0, 0], [0, 1], c='k', lw=1)
    plot.plot([0.25, 0.25], [0, 1], 'k--', lw=1)
    plot.plot([0.5, 0.5], [0, 1], 'k--', lw=1)
    plot.plot([0.75, 0.75], [0, 1], 'k--', lw=1)
    # plot.plot([1, 1], [0, 1], c='k', lw=1)
    # plot.plot([0, 1], [0, 0], c='k', lw=1)
    plot.plot([0, 1], [0.25, 0.25], 'k--', lw=1)
    plot.plot([0, 1], [0.5, 0.5], 'k--', lw=1)
    plot.plot([0, 1], [0.75, 0.75], 'k--', lw=1)
    # plot.plot([0, 1], [1, 1], c='k', lw=1)

    plot.set_xlim((0, 1))
    plot.set_ylim((0, 1))
    plot.axis('off')


def plot_difference_IFS(plot, data, bin_frac):
    """Plot differences between successive values in a time series along with IFS bins"""
    """"""
    """Inputs:"""
    """plot -- matplotlib plot"""
    """data -- 1d numpy array of differences"""
    """bin_frac -- bin_frac used in driven IFS analysis"""

    plot.scatter(np.arange(len(data)), data, marker='.', s=0.5)
    xlim = plot.get_xlim()
    mx = np.max(data)
    mn = np.min(data)
    plot.plot(xlim, [-bin_frac*abs(mx-mn), -bin_frac*abs(mx-mn)], 'k--')
    plot.plot(xlim, [0, 0], 'k--')
    plot.plot(xlim, [bin_frac*abs(mx-mn), bin_frac*abs(mx-mn)], 'k--')
    plot.axis('off')


def change_hist(plot, data):
    """Draws histogram for multiple price changes"""

    # Plot histogram
    numbins = 7
    plot.hist(data, bins=numbins, normed=False, color='black', alpha=0.2, \
                   histtype='stepfilled', label=r'data')

    # Plot 0 line
    plot.plot([0, 0], plot.get_ylim(), 'r--', lw=1)

    # Plot mean line
    mu = np.mean(data)
    plot.plot([mu, mu], plot.get_ylim(), 'k--', lw=1)

    # Stylistic elements
    plot.axis('off')
    # if abs(np.min(data)) > abs(np.max(data)):
    #     xscale = [np.min(data), -np.min(data)]
    # else:
    #     xscale = [-np.max(data), np.max(data)]
    # xscale = [xscale[0]-.25*(xscale[1]-xscale[0]), xscale[1]+.25*(xscale[1]-xscale[0])]
    # plot.set_xlim(xscale)
    # plot.annotate(r'%s' % round(np.min(data), 2), xy=(0, -0.06), textcoords='axes fraction', fontsize=10)
    # plot.annotate(r'%s' % round(np.max(data), 2), xy=(1, -0.06), textcoords='axes fraction', fontsize=10)


def range_hist(plot, data):
    """Draws histogram for data given certain timeframe"""

    # Plot histogram
    numbins = 50
    plot.hist(data, bins=numbins, normed=False, color='black', alpha=0.2, \
                   histtype='stepfilled', label=r'data')

    # Plot mean line

    mu = np.mean(data)
    plot.plot([mu, mu], plot.get_ylim(), 'k--', lw=1)

    # Stylistic elements
    xscale = plot.get_xlim()
    xscale = [xscale[0]-.25*(xscale[1]-xscale[0]), xscale[1]+.25*(xscale[1]-xscale[0])]
    plot.set_xlim(xscale)
    plot.axis('off')


def plot_density(plot, x_plot, function, ppos, pneg):
    """Plot density function and probabilities"""

    plot.axis('off')
    plot.fill_between(x_plot, function, where=x_plot<0, interpolate=True, color='red', alpha=0.1)
    plot.fill_between(x_plot, function, where=x_plot>0, interpolate=True, color='black', alpha=0.45)

    plot.annotate(r'$\Pr[X < 0] = %s$' %round(ppos[0]*100,2) + r'$\%$', xycoords='axes fraction', xy=(0.68, 0.88), size=9, horizontalalignment='left', color='k')
    plot.annotate(r'$\Pr[X > 0] = %s$' %round(pneg[0]*100,2) + r'$\%$', xycoords='axes fraction', xy=(0.05, 0.88), size=9, horizontalalignment='left', color='k')


def lagplot(plot, data, lag):
    dlag = data[lag:]
    plot.scatter(data[:len(data)-lag], dlag, marker='.', s=0.5)
    plot.set_title('lagplot, l=%s' % (str(lag)))


def return_period_memory(plot, rdata, max_timeframe):
    """
    http://www.bearcave.com/misl/misl_tech/wavelets/hurst/index.html#RSAndFinance
    """
    data = []
    for period in np.arange(max_timeframe):
        returns = np.array([r2-r1 for r2, r1 in zip(rdata[period:], rdata)])
        hu = hurst(returns)
        data.append([period, hu])
    data = np.array(data)
    plot.scatter(data[:, 0], data[:, 1], marker='D', s=1.5, color='b')
    plot.plot(data[:, 0], data[:, 1], 'k')
    plot.set_xlabel('return period')
    plot.set_ylabel('hurst')
    plot.set_title('return period memory')


def autocorrelation_plot(plot, data, max_lag):

    y = [autocorrelation(data, lag) for lag in range(max_lag)]
    plot.bar(np.arange(max_lag), y)
    plot.set_title('autocorrelations')
    plot.set_xlabel('autocorrelation lag')
    plot.set_ylabel('correlation')
