import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats

def pot_scale_factor(this_pot, target = 1e20):
    
    return target / this_pot

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def efficiency(num, den, num_w=None, den_w=None, n_bins=10, limits=None, conf_level=None):
    '''
    Calculates the efficiency given two populations: one containig 
    the totatility of the events, and one containing only events 
    that pass the selection.
    It uses a frequentist approach to evaluate the uncertainty.
    Other methods are to be implemented.
    
    Arguments:
        num {tuple} -- The totality of the events
        den {tuple} -- The events that pass the selection
        num_w {tuple} -- Optional, the weight for every event
        den_w {tuple} -- Optional, the weight for every selected event
        n_bins {int} -- Optional, the number of bins
        limits {tuple} -- Optional, the lower and upper limits of the bins 
        conf_level {float} -- Optional, the confidence level to be used
        
    Outputs:
        eff {tuple} -- The efficiency per bin
        unc_low {tuple} -- The lower uncertainty per bin
        unc_up {tuple} -- The upper uncertainty per bin
        bins {tuple} -- The bin edges
        bins_mid {tuple} -- The mid points of the bins
        x_bar {tuple} -- The uncertainty along the x axis
    '''
    
    if num_w is None:
        num_w = [1.] * len(num)
        
    if den_w is None:
        den_w = [1.] * len(den)
        
    if conf_level is None:
        conf_level = 0.682689492137
        
    if limits is None:
        x_min=0
        x_max=10
    else:
        x_min=limits[0]
        x_max=limits[1]

    num = np.asarray(num, dtype=np.float32)
    num_w = np.asarray(num_w, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    den_w = np.asarray(den_w, dtype=np.float32)

    bins = np.linspace(x_min, x_max, n_bins)

    num_h, _    = np.histogram(num, bins=bins)
    num_w_h, _  = np.histogram(num, weights=num_w, bins=bins)
    num_w2_h, _ = np.histogram(num, weights=num_w**2, bins=bins)

    den_h, _    = np.histogram(den, bins=bins)
    den_w_h, _  = np.histogram(den, weights=den_w, bins=bins)
    den_w2_h, _ = np.histogram(den, weights=den_w**2, bins=bins)

    eff = num_w_h / den_w_h

    variance = (num_w2_h * (1. - 2 * eff) + den_w2_h * eff *eff ) / ( den_w_h * den_w_h)
    sigma = np.sqrt(variance)
    prob = 0.5 * (1. - conf_level)
    delta = - scipy.stats.norm.ppf(prob) * sigma

    unc_up = []
    unc_low = []

    for eff_i, delta_i in zip(eff, delta):
        if eff_i - delta_i < 0:
            unc_low.append(eff_i)
        else:
            unc_low.append(delta_i)
            
        if eff_i + delta_i > 1:
            unc_up.append(1. - eff_i)
        else:
            unc_up.append(delta_i)
            
            
    bins_mid = [bins[i]+(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)]
    x_bar = 0.5*(bins[-1]-bins[0])/len(bins)
    
    return eff, unc_low, unc_up, bins, bins_mid, x_bar


def plot_histogram(ax, data, option='simple', weights=None, n_bins=10, limits=None, label=None, hcolors=None):
    '''
    Plots a histogram given values.
    
    Arguments:
        ax {ax} -- The axes to use for plotting
        data {tuple} -- The data to be histogrammed
        option {str} -- Optional, the option for plotting
        weights {tuple} -- Optional, The weights
        n_bins {int} -- Optional, the number of bins
        limits {tuple} -- Optional, the lower and upper limits of the bins
        label {tuple} -- Optional, the labels
        hcolors {tuple} -- Optional, the colors
        
    Options:
        simple -- With filled error bars
        stacked -- Stacked histograms
    '''
    
    data_h, err, bins, bins_mid = histogram_helper(data, weights, n_bins, limits)
    
    widths = bins[1:] - bins[:-1]

    loopable_data_h = []
    loopable_err = []
    total_events = 0
    if isinstance(data_h[0], list) or isinstance(data_h[0], np.ndarray):
        loopable_data_h = data_h
        loopable_err = err
        for d in data_h: total_events += np.sum(d)
    else:
        loopable_data_h.append(data_h)
        loopable_err.append(err)
        total_events = np.sum(data_h)
        
    if label is None:
        label = ['empty'] * len(data)
    else:
        if len(label) != len(data):
            raise ('Length of data and label has to be the same.')
            
    if hcolors is None:
        hcolors = colors 
        
    lower = np.zeros(len(loopable_data_h[0]))
    
    totals = []
        
    for d, e, l in zip(loopable_data_h, loopable_err, label):
        
        totals.append(np.sum(d))
        n_events = np.sum(d)
                
        d = np.append(d, d[-1])
        e = np.append(e, e[-1])
        
        if option == 'simple':
            ax.step(
                bins,
                d,
                color=hcolors[len(totals)-1],
                where="post",
                label=l,
            )
            ax.fill_between(
                bins, d - e, d + e, alpha=0.3, step="post", 
                color=hcolors[len(totals)-1]
            )
    
    
        elif option == 'stacked':
            ax.bar(bins_mid,
                    d[:-1],
                    lw=2,
                    width=widths,
                    bottom=lower,
                    label=l+', {0:0.1f}%'.format(n_events / total_events * 100),
                    color=hcolors[len(totals)-1])
            lower += d[:-1]
        else:
            raise('Option' + option + 'not recognized.')

    # Uncertainty bars for stacked histograms
    if option == 'stacked':
        
        # Unify all the data, together with their weights
        if isinstance(data[0], list) or isinstance(data_h[0], np.ndarray):            
            d = data[0]
            w = weights[0]
            for i in range(1, len(data)):
                d = np.append(d, data[i])
                w = np.append(w, weights[i])
        else:
            d = data
            w = weights
        
        # Make a histogram of the total data
        data_tot, err_tot, _, _ = histogram_helper(d, w, n_bins, limits)
        
        # Plot the histogram as a hashed histogram
        for m, v, e, w in zip(bins_mid, data_tot, err_tot, widths):
            ax.add_patch(
                    patches.Rectangle(
                    (m - w / 2, v - e),
                    w,
                    2 * e,
                    hatch="\\\\\\\\\\",
                    Fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )
    return totals
    
    
def histogram_helper(data, weights=None, n_bins=10, limits=None):
    '''
    Makes a histogram of values.
    
    Arguments:
        data {tuple} -- The data to be histogrammed
        weights {tuple} -- Optional, The weights
        n_bins {int} -- Optional, the number of bins
        limits {tuple} -- Optional, the lower and upper range of the bins 
        
    Outputs:
        data_h {tuple} -- The bin content per bin
        err {tuple} -- The statistical uncertainty per bin
        bins {tuple} -- The bin edges
        bins_mid {tuple} -- The mid points of the bins
    '''
    
    if limits is None:
        x_min=0
        x_max=10
    else:
        x_min=limits[0]
        x_max=limits[1]
        
#     if weights != 0 and len(data) != len(weights):
#         raise ('Data and weights must have the same length.')
        
    bins = np.linspace(x_min, x_max, n_bins+1)
    bins_mid = [bins[i]+(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)]
        
    loopable_data = []
    loopable_weights = []
    if isinstance(data, list):
        loopable_data = data
        loopable_weights = weights
    else:
        loopable_data.append(data)
        loopable_weights.append(weights)
    
    data_h = []
    err = []
    
    for d, w in zip(loopable_data, loopable_weights):
        
        d = np.asarray(d, dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)
    
        if w is None:
            h, _ = np.histogram(d, bins=bins)
            e    = np.sqrt(h)
        else:
            h, _    = np.histogram(d, weights=w, bins=bins)
            w2_h, _ = np.histogram(d, weights=w**2, bins=bins)
            e = np.sqrt(w2_h)
            
        data_h.append(h)
        err.append(e)
        
    if isinstance(data, list):
        return data_h, err, bins, bins_mid
    else:
        return data_h[0], err[0], bins, bins_mid