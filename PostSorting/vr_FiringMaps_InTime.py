import warnings
import numpy as np
import matplotlib.pylab as plt
import scipy.stats
import gc

def nextpow2(x):
    """ Return the smallest integral power of 2 that >= x """
    n = 2
    while n < x:
        n = 2 * n
    return n


def fftkernel(x, w):
    """
    y = fftkernel(x,w)
    Function `fftkernel' applies the Gauss kernel smoother to an input
    signal using FFT algorithm.
    Input argument
    x:    Sample signal vector.
    w: 	Kernel bandwidth (the standard deviation) in unit of
    the sampling resolution of x.
    Output argument
    y: 	Smoothed signal.
    MAY 5/23, 2012 Author Hideaki Shimazaki
    RIKEN Brain Science Insitute
    http://2000.jukuin.keio.ac.jp/shimazaki
    Ported to Python: Subhasis Ray, NCBS. Tue Jun 10 10:42:38 IST 2014
    """
    L = len(x)
    Lmax = L + 3 * w
    n = nextpow2(Lmax)
    X = np.fft.fft(x, n)
    f = np.arange(0, n, 1.0) / n
    f = np.concatenate((-f[:int(n / 2)], f[int(n / 2):0:-1]))
    K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)
    y = np.fft.ifft(X * K, n)
    y = y[:L].copy()
    return y


"""
## Here we convolve spikes in time trial by trial basis on clusters
1. load data:
    data is firing times in ms
2. bin spikes into 250 ms windows
"""


def generate_time_bins(spike_times):
    time_bins = np.arange(0,np.max(spike_times),7500)
    return time_bins


def generate_time_bins_for_speed(speed):
    time_bins = np.arange(0, (speed.shape[0]), 7500)
    return time_bins


def bin_spike_times(spike_times, number_of_bins):
    binned_spikes_in_time = create_histogram(spike_times, number_of_bins)
    return binned_spikes_in_time


def create_histogram(spike_times, number_of_bins):
    posrange = np.linspace(number_of_bins.min(), number_of_bins.max(),  num=number_of_bins.shape[0])
    values = np.array([posrange[0], posrange[-1]])
    H, bins = np.histogram(spike_times, bins=(posrange), range=values)
    return H


def convolve_binned_spikes(binned_spike_times):
    convolved_spikes=[]
    convolved_spikes = fftkernel(binned_spike_times, 2)
    return convolved_spikes


def bin_spatial_array(array, bins):
    binned_array=[]
    bin_len = int(bins.shape[0])-1
    for bcount, bin in enumerate(bins[:bin_len]):
        next_bin=bin+7500
        mean = np.nanmean(np.array(array[bin:next_bin]), dtype=np.float)
        binned_array = np.append(binned_array, mean)
    return binned_array


def convolve_spikes_in_time(spike_data, cluster_index, number_of_bins):
    gc.collect()
    spike_times = np.array(spike_data.firing_times.iloc[cluster_index])
    binned_spike_times = bin_spike_times(spike_times, number_of_bins)
    convolved_spikes = convolve_binned_spikes(binned_spike_times)
    spike_data.spike_rate_in_time.iloc[cluster_index] = convolved_spikes
    return spike_data


def convolve_speed_in_time(spike_data, cluster_index,raw_spatial_data):
    speed = np.array(raw_spatial_data["speed_per200ms"])
    number_of_bins = generate_time_bins_for_speed(speed)
    binned_speed = bin_spatial_array(speed, number_of_bins)
    convolved_speed = convolve_binned_spikes(binned_speed)
    spike_data.speed_rate_in_time.iloc[cluster_index] = convolved_speed
    return spike_data, number_of_bins


def convolve_position_in_time(spike_data, cluster_index,raw_spatial_data, number_of_bins):
    location = np.array(raw_spatial_data["x_position_cm"])
    binned_position = bin_spatial_array(location, number_of_bins)
    spike_data.position_rate_in_time.iloc[cluster_index] = binned_position
    return spike_data



def control_convolution_in_time(spike_data,raw_spatial_data):
    spike_data["spike_rate_in_time"] = ""
    spike_data["speed_rate_in_time"] = ""
    spike_data["position_rate_in_time"] = ""

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        spike_data, number_of_bins = convolve_speed_in_time(spike_data, cluster_index,raw_spatial_data)
        spike_data = convolve_spikes_in_time(spike_data, cluster_index,number_of_bins)
        spike_data = convolve_position_in_time(spike_data, cluster_index,raw_spatial_data, number_of_bins)
    return spike_data