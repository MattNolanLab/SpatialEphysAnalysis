import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pylab as plt
import math
import os
import pandas as pd
import PostSorting.parameters
import pickle
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import traceback
import warnings
import sys
from scipy import fftpack
import elephant
warnings.filterwarnings('ignore')

test_params = PostSorting.parameters.Parameters()
import elephant as elephant

"""
https://elifesciences.org/articles/35949#s4
eLife 2018;7:e35949 DOI: 10.7554/eLife.35949
Kornienko et al., 2018
The theta rhythmicity of neurons was estimated from the instantaneous firing rate of the cell.
The number of spikes observed in 1 ms time window was calculated and convolved with a Gaussian kernel (standard deviation of 5 ms).
The firing probability was integrated over 2 ms windows and transformed into a firing rate.
A power spectrum of the instantaneous firing rate was calculated using the pwelchfunction of the oce R package.
The estimates of the spectral density were scaled by multiplying them by the corresponding frequencies: spec(x)∗freq(x).
A theta rhythmicity index for each neuron was defined as θ−baselineθ+baseline, where θ is the mean power at 6–10 Hz and baseline is the mean power in two adjacent frequency bands (3–5 and 11–13 Hz).
The theta rhythmicity indices of HD cells were analyzed with the Mclust function of the R package mclust which uses Gaussian mixture modeling and the EM algorithm to estimate the number of components in the data.
# obtain spike-time autocorrelations
windowSize<-300; binSize<-2
source(paste(indir,"get_stime_autocorrelation.R",sep="/"))
runOnSessionList(ep,sessionList=rss,fnct=get_stime_autocorrelation,
                 save=T,overwrite=T,parallel=T,cluster=cl,windowSize,binSize)
rm(get_stime_autocorrelation)
get_frequency_spectrum<-function(rs){
  print(rs@session)
  myList<-getRecSessionObjects(rs)
  st<-myList$st
  pt<-myList$pt
  cg<-myList$cg
  wf=c();wfid=c()
  m<-getIntervalsAtSpeed(pt,5,100)
  for (cc in 1:length(cg@id)) {
    st<-myList$st
    st<-setCellList(st,cc+1)
  ##########################################################
    st1<-setIntervals(st,s=m)
    st1<-ifr(st1,kernelSdMs = 5,windowSizeMs = 2)
    Fs=1000/2
    x=st1@ifr[1,]
    xts <- ts(x, frequency=Fs)
    w <- oce::pwelch(xts,nfft=512*2, plot=FALSE,log="no")
    wf0=w$spec*w$freq
    wf=rbind(wf,wf0)
    wfid=cbind(wfid,cg@id[cc])
  }
  return(list(spectrum=t(wf),spectrum.id=wfid,spectrum.freq=w$freq))
  }
  
##################################################################################
# calculate theta index from power spectra of instantaneous firing rates
freq=spectrum.freq[1,]
theta.i=c()
for (i in 1:dim(spectrum)[2]){
  wf=spectrum[,i]
  th=mean(wf[freq>6 & freq<10])
  b=mean(c(wf[freq>3 & freq<5],wf[freq>11 & freq<13]))
  ti=(th-b)/(th+b)
  theta.i=c(theta.i,ti)
}
x=theta.i[t$hd==1]
par(mfrow=c(2,3))
hist(x,main="Theta index distribution",ylab = "Number of cells", xlab="Theta index",15,xlim = c(-.05,.4),las=1)
x.gmm = Mclust(x)
x.s=summary(x.gmm)
print("Fit Gaussian finite mixture model")
print(paste("Number of components of best fit: ",x.s$G,sep=""))
print(paste("Log-likelhood: ",round(x.s$loglik,2),sep=""))
print(paste("BIC: ",round(x.s$bic,2),sep=""))
print("Theta index threshold = 0.07")
lines(c(0.07,0.07),c(0,14),col="red",lwd=2)
print(paste("Number of non-rhythmic (NR) HD cells (theta index threshold < 0.07): N = ",sum(x<.07),sep=""))
print(paste("Number of theta-rhythmic (TR) HD cells (theta index threshold > 0.07): N = ",sum(x>.07),sep=""))
##################################################################################
  
"""


"""
Alternative Theta modulation classification. 
Boccara et al. Nature Neuroscience, 2010): "Individual cells were defined as
being theta modulated if the mean spectral power within 1 Hz of the peak in 
the 4–11-Hz frequency range of the spike-train autocorrelogram was at least 
fivefold greater than the mean spectral power between 0 and 125 Hz

This is calculated and called Boccara_theta_class either 1=modulated, 0=not modulated
"""

def calculate_spectral_density(firing_rate, cluster_data, save_path):
    f, Pxx_den = signal.welch(firing_rate, fs=1000, nperseg=10000, scaling='spectrum')
    Pxx_den = Pxx_den*f
    #print(cluster)
    #plt.semilogy(f, Pxx_den)
    plt.plot(f, Pxx_den, color='black')
    plt.xlim(([0,20]))
    plt.ylim([0,max(Pxx_den)])
    #plt.ylim([1e1, 1e7])
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    try:
        plt.savefig(save_path + '/' + cluster_data.session_id + '_' + str(cluster_data.cluster_id) + '_power_spectra_ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
    except:
        print("you do not have permission to save a figure here, ", save_path)
    plt.close()
    return f, Pxx_den


def calculate_firing_probability(convolved_spikes):
    firing_rate=[]
    firing_rate = get_rolling_sum(convolved_spikes, 2)
    return (firing_rate*1000)/2 # convert to Hz


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:] / window


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window is too big, plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

def extract_instantaneous_firing_rate(cluster_data):
    firing_times=cluster_data.firing_times/30 # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times), 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    smoothened_instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)
    return smoothened_instantaneous_firing_rate

def calculate_theta_power(Pxx_den,f):
    theta_power = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f > 6, f < 10))))
    #baseline = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >  0, f < 50))))
    adjacent_power1 = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >= 3, f <=5))))
    adjacent_power2 = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >= 11, f <=13))))
    baseline = (adjacent_power1 + adjacent_power2)/2
    x = theta_power - baseline
    y = theta_power + baseline
    t_index = x/y
    return t_index, theta_power

def calculate_boccara_theta(Pxx_den, f):
    # see above for definition


    f_peak = calculate_theta_peak(Pxx_den, f)
    f_lower = f_peak-1
    f_higher = f_peak+1
    mean_peak_theta_power = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f > f_lower, f < f_higher))))
    mean_baseline_power = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f > 0, f < 125))))

    if mean_peak_theta_power>(mean_baseline_power*5):
        boccara_theta_mod = 1 # classified as theta modulated if at least 5 fold larger
    else:
        boccara_theta_mod = 0

    #print(mean_peak_theta_power/mean_baseline_power)
    return boccara_theta_mod

def calculate_boccara_theta_2(firing_rate, sampling_rate, cluster_data):

    firing_times_cluster = cluster_data.firing_times
    corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/sampling_rate, 1, 2000)

    #corr, time = calculate_autocorrelogram_hist2(np.array(firing_times_cluster), 1, prm.get_sampling_rate())
    #time = time/prm.get_sampling_rate()
    corr = corr[1000:]
    time = time[1000:]
    '''
    fig, ax = plt.subplots()
    ax.plot(corr)
    plt.savefig("/mnt/datastore/Harry/signal.png")

    f = 7  # Frequency, in cycles per second, or Hertz
    f_s = 100  # Sampling rate, or number of measurements per second
    t = np.linspace(0, 2, 2 * f_s, endpoint=False)
    x = np.sin(f * 2 * np.pi * t)
    fig, ax = plt.subplots()
    ax.plot(x)
    plt.savefig("/mnt/datastore/Harry/sim_signal.png")

    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * f_s
    freqs, X = signal.welch(x, f_s)
    fig, ax = plt.subplots()
    ax.plot(freqs, np.abs(X))
    ax.set_xlim([0,50])
    plt.savefig("/mnt/datastore/Harry/sim_fft_signal.png")
    
    '''

    fig = plt.figure(figsize=(7,4)) # width, height?
    ax = fig.add_subplot(1, 2, 1)  # specify (nrows, ncols, axnum)
    ax.plot(time, corr)

    ax = fig.add_subplot(1, 2, 2)  # specify (nrows, ncols, axnum)
    '''
    corr_fft = fftpack.rfft(corr)
    freqs = (fftpack.rfftfreq(len(corr_fft))*1000)
    corr_fft = corr_fft[1::2]
    freqs = freqs[1::2]
    corr_fft = np.abs(corr_fft)**2
    ax.plot(freqs, corr_fft, label="fft")
    ax.set_xlim([0, 125])
    '''
    freqs, corr_fft = signal.welch(corr, fs=1000, scaling='density')
    freqs, corr_fft = elephant.spectral.welch_psd(corr, freq_res=1, fs=1000)
    freqs = freqs[1:]
    corr_fft = corr_fft[1:]
    corr_fft = corr_fft/np.sum(corr_fft)
    ax.plot(freqs, corr_fft, label="fft")
    ax.set_xlim([0, 125])

    f_peak = calculate_theta_peak(corr_fft, freqs)
    f_lower = f_peak-1
    f_higher = f_peak+1
    mean_peak_theta_power = np.nanmean(np.take(corr_fft, np.where(np.logical_and(freqs >= f_lower, freqs <= f_higher))))
    mean_baseline_power = np.nanmean(np.take(corr_fft, np.where(np.logical_and(freqs >= 0, freqs <= 125))))

    if mean_peak_theta_power>(mean_baseline_power*5):
        boccara_theta_mod = 1 # classified as theta modulated if at least 5 fold larger
    else:
        boccara_theta_mod = 0
        
    return boccara_theta_mod



def calculate_theta_peak(Pxx_den,f):
    theta_powers = np.take(Pxx_den, np.where(np.logical_and(f >= 4, f <=11)))
    theta_frequencies = np.take(f, np.where(np.logical_and(f >= 4, f <=11)))
    theta_peak = theta_frequencies[0, np.argmax(theta_powers)]
    return theta_peak

def calculate_theta_index(spike_data, output_path, sampling_rate):
    print('I am calculating theta index...')
    save_path = output_path + '/Figures/firing_properties/autocorrelograms'
    try:
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
    except:
        print("you do not have permissioned to create a directory here, ", save_path)


    theta_indices = []
    theta_powers = []
    boccara_thetas = []

    for cluster in range(len(spike_data)):
        cluster_data = spike_data.iloc[cluster]

        if len(cluster_data.firing_times)<=1:
            # in the case no or 1 spike is found in open field or vr
            theta_indices.append(np.nan)
            theta_powers.append(np.nan)
            boccara_thetas.append(np.nan)

        else:
            instantaneous_firing_rate = extract_instantaneous_firing_rate(cluster_data)
            firing_rate = calculate_firing_probability(instantaneous_firing_rate)
            f, Pxx_den = calculate_spectral_density(firing_rate, cluster_data, save_path)
            #boccara_theta = calculate_boccara_theta(Pxx_den, f)
            boccara_theta = calculate_boccara_theta_2(firing_rate, sampling_rate, cluster_data)
            t_index, t_power = calculate_theta_power(Pxx_den, f)

            theta_indices.append(t_index)
            theta_powers.append(t_power)
            boccara_thetas.append(boccara_theta)

            #print("t_index = "+str(t_index))
            #print("theta_powers = "+str(t_power))
            #print("boccara_thetas = "+str(boccara_theta))

            firing_times_cluster = cluster_data.firing_times
            corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/sampling_rate, 1, 600)

            fig = plt.figure(figsize=(7,4)) # width, height?
            ax = fig.add_subplot(1, 2, 1)  # specify (nrows, ncols, axnum)
            ax.set_ylim(bottom=0, top=max(corr)+(0.05*max(corr)))
            ax.set_ylabel("Spike prob.", fontsize=15)
            ax.set_xlabel("Time (ms)", fontsize=15)
            ax.set_xlim(-300, 300)
            ax.plot(time, corr, '-', color='black')
            x=np.max(corr)
            #ax.text(-200,x, "theta index = " + str(round(t_index,3)), fontsize =10)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            ax = fig.add_subplot(1, 2, 2)  # specify (nrows, ncols, axnum)
            ax.plot(f, Pxx_den, color='black')
            plt.ylim([0,max(Pxx_den)])
            plt.xlim(([0,20]))
            ax.set_xlabel('Frequency (Hz)', fontsize=15)
            ax.set_ylabel('Power', fontsize=15)
            x = max(Pxx_den)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            ax.text(1,x, "Theta index = " + str(np.round(t_index,decimals=2)), fontsize =15)
            fig.tight_layout(pad=3.0)
            try:
                plt.savefig(save_path + '/' + cluster_data.session_id + '_' + str(cluster_data.cluster_id) + '_theta_properties.png', dpi=300)
            except:
                print("you do not have permission to save a figure here, ", save_path)
            plt.close()

    spike_data["ThetaPower"] = theta_powers
    spike_data["ThetaIndex"] = theta_indices
    spike_data["Boccara_theta_class"] = boccara_thetas

    return spike_data


def calculate_autocorrelogram_hist(spikes, bin_size, window):

    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time

def calculate_autocorrelogram_hist2(spikes, bin_size, window):

    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike])
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time



def convolve_array(spikes):
    window = signal.gaussian(2, std=5)
    convolved_spikes = signal.convolve(spikes, window, mode='full')
    return convolved_spikes


#A theta rhythmicity index for each neuron was defined as θ−baselineθ+baseline,
## where θ is the mean power at 6–10 Hz and baseline is the mean power in two adjacent frequency bands (3–5 and 11–13 Hz).

"""
The theta index was calculated here as by Yartsev et al., (2011). 
First, we computed the autocorrelation of the spike train binned by 0.01 seconds with lags up to ±0.5 seconds. 
Without normalization, this may be interpreted as the counts of spikes that occurred in each 0.01 second bin 
after a previous spike (Figure 1a). The mean was then subtracted, and the spectrum was calculated as the square 
of the magnitude of the fast-Fourier transform of this signal, zero-padded to 216 samples. 
This spectrum was then smoothed with a 2-Hz rectangular window (Figure 1b), 
and the theta index was calculated as the ratio of the mean of the spectrum within 1-Hz of each side of the 
peak in the 5-11 Hz range to the mean power between 0 and 50 Hz.
"""

def run_test(spatial_firing, id=None):
    if id is not None:
        spatial_firing = spatial_firing[spatial_firing["cluster_id"] == id]

    if "ThetaIndex" in list(spatial_firing):
        spatial_firing = spatial_firing.drop(columns=["ThetaIndex"])
    if "ThetaPower" in list(spatial_firing):
        spatial_firing = spatial_firing.drop(columns=["ThetaPower"])

    spatial_firing = calculate_theta_index(spatial_firing, test_params)
    return spatial_firing

def gen_random_firing(n_clusters):
    print("done")
    spatial_firing = pd.DataFrame()
    cluster_ids = []
    firing_times = []
    session_ids = []
    for i in range(n_clusters):
        cluster_ids.append(i+1)
        firing_times.append(np.sort(np.random.choice(100000000, 10000, replace=False)))
        session_ids.append("random_firing_test")
    spatial_firing["firing_times"] = firing_times
    spatial_firing["cluster_id"] = cluster_ids
    spatial_firing["session_id"] = session_ids
    return spatial_firing

def gen_modulated_firing(n_clusters, freq=8):
    print("done")

    total_length = 1000000
    n_spikes = 10000
    F = freq # desired frequnecy
    Fs = 30000 # sampling rate
    T = total_length/Fs # n seconds   30000
    Ts = 1./Fs
    N = int(T/Ts)
    t = np.linspace(0, T, N)
    signal = np.cos(2*np.pi*F*t)+1
    signal_normalised = signal/np.sum(signal)

    spatial_firing = pd.DataFrame()
    cluster_ids = []
    firing_times = []
    session_ids = []
    for i in range(n_clusters):
        cluster_ids.append(i+1)
        firing_times.append(np.sort(np.random.choice(total_length, n_spikes, replace=False, p=signal_normalised)))
        session_ids.append("random_firing_test")
    spatial_firing["firing_times"] = firing_times
    spatial_firing["cluster_id"] = cluster_ids
    spatial_firing["session_id"] = session_ids
    return spatial_firing

def run_for_x(path_to_recording):
    list_of_recordings = [f.path for f in os.scandir(path_to_recording) if f.is_dir()]

    for i in range(len(list_of_recordings)):
        try:
            recording_path = list_of_recordings[i]
            print("processing recording ", recording_path)
            spatial_firing_path = recording_path + "/MountainSort/DataFrames/spatial_firing.pkl"
            spatial_firing = pd.read_pickle(spatial_firing_path)
            spatial_firing= spatial_firing.sort_values(by=['cluster_id'])
            test_params.set_output_path(recording_path+"/MountainSort")
            test_params.set_sampling_rate(30000)
            spatial_firing = run_test(spatial_firing)
            spatial_firing.to_pickle(spatial_firing_path)
            print("successful on recording, ", list_of_recordings[i])

        except Exception as ex:
            print("failed on recording, ", list_of_recordings[i])
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

def gen_modulated_firing_figure(freq, save_path):

    total_length = 1000000
    n_spikes = 10000
    F = freq # desired frequnecy
    Fs = 30000 # sampling rate
    T = total_length/Fs # n seconds   30000
    Ts = 1./Fs
    N = int(T/Ts)
    t = np.linspace(0, T, N)

    signal1 = np.cos(2*np.pi*F*t)+1
    signal1 += np.random.normal(0, 0.3, len(signal1))

    #signal1_normalised = signal1/np.sum(signal1)
    #signal1_normalised += np.random.normal(0, 0, len(signal1_normalised))
    #signal2 = np.cos(2*np.pi*1234*t)+1
    #signal2_normalised = signal2/np.sum(signal2)
    #signal2_normalised = 0

    #signal_total = signal1_normalised+signal2_normalised

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(signal1[0:32000], color="black")
    plt.savefig(save_path+"/mod_firing_example.png", dpi=300)
    print("plotted depth correlation")


