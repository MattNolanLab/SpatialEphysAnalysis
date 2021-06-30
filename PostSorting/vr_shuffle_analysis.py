# Import packages
import numpy as np
from Functions_Params_0100 import STOP_THRESHOLD, DIST, HDF_LENGTH, BINNR, SHUFFLE_N
from scipy.stats import uniform
import random


def create_srdata(stops, trialids):
    if stops.size == 0:
        return np.zeros((BINNR,))

    # create histogram
    posrange = np.linspace(0, HDF_LENGTH, num=BINNR + 1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1] + 1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],
                       [posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:, 2], stops[:, 0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::] > 1)] = 1

    return H


def shuffle_stops2( stops,n ):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    # create an array that contains the amount by which every stop will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=HDF_LENGTH, size=stops.shape[0])
    # add random value
    shuffled_stops[:,0] = rand_rotation
    shuffled_stops[:,2] = n


# create shuffled and real datasets
def calculate_shuffled_spikes(stopsdata, trialids):
    SHUFFLE1 = 100
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    # Shuffling random 100 trials 1000 times
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, BINNR))
    for i in range(SHUFFLE_N): # for i in 1000
        shuffledtrials = np.zeros((SHUFFLE1, 5))
        shuffleddata =np.zeros((SHUFFLE1, BINNR))
        for n in range(SHUFFLE1): # Create sample data with 100 trials
            trial = random.choice(trialids) # select random trial from real dataset
            data = stopsdata[stopsdata[:,2] ==trial,:] # get data only for each tria
            shuffledtrial = shuffle_stops2(data,n) # shuffle the locations of stops in the trial
            shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stop
        trialids2 = np.unique(shuffledtrials[:, 2]) # find unique trials in the data
        shuffled_srbin = create_srdata( shuffledtrials, trialids2 ) #
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
    # Mean of the mean stops in the shuffled data for each bin

    return srbin, shuffled_srbin_mean


def generate_shuffled_data(spike_data):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        cluster_firing_indices = spike_data.firing_times[cluster_index]
