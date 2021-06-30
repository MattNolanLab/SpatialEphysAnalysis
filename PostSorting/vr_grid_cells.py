import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
from scipy import stats
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import os
import traceback
import warnings
import sys
warnings.filterwarnings('ignore')


def calculate_grid_field_com(cluster_spike_data, position_data, bin_size, prm):
    '''
    :param spike_data:
    :param prm:
    :return:

    for each trial of each trial type we want to
    calculate the centre of mass of all detected field
    centre of mass is defined as

    '''

    firing_field_com = []
    firing_field_com_trial_numbers = []
    firing_field_com_trial_types = []
    firing_rate_maps = []

    #firing_field = []

    firing_times=cluster_spike_data.firing_times/(prm.get_sampling_rate()/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    if len(firing_times)==0:
        firing_rate_maps = np.zeros(int(prm.track_length))
        return firing_field_com, firing_field_com_trial_numbers, firing_field_com_trial_types, firing_rate_maps

    trial_numbers = np.array(position_data['trial_number'].to_numpy())
    trial_types = np.array(position_data['trial_type'].to_numpy())
    time_seconds = np.array(position_data['time_seconds'].to_numpy())
    x_position_cm = np.array(position_data['x_position_cm'].to_numpy())

    instantaneous_firing_rate_per_ms = extract_instantaneous_firing_rate_for_spike2(cluster_spike_data, prm) # returns firing rate per millisecond time bin
    instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[0:len(x_position_cm)]

    if not (len(instantaneous_firing_rate_per_ms) == len(trial_numbers)):
        # 0 pad until it is the same size (padding with 0 hz firing rate
        instantaneous_firing_rate_per_ms = np.append(instantaneous_firing_rate_per_ms, np.zeros(len(trial_numbers)-len(instantaneous_firing_rate_per_ms)))

    for trial_number in np.unique(trial_numbers):
        trial_type = stats.mode(trial_types[trial_numbers==trial_number])[0][0]
        trial_x_position_cm = x_position_cm[trial_numbers==trial_number]
        trial_instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[trial_numbers==trial_number]

        numerator, bin_edges = np.histogram(trial_x_position_cm, bins=int(prm.get_track_length()/bin_size), range=(0, prm.track_length), weights=trial_instantaneous_firing_rate_per_ms)
        denominator, bin_edges = np.histogram(trial_x_position_cm, bins=int(prm.get_track_length()/bin_size), range=(0, prm.track_length))
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

        firing_rate_map = numerator/denominator

        local_maxima_bin_idx = signal.argrelextrema(firing_rate_map, np.greater)[0]
        global_maxima_bin_idx = np.nanargmax(firing_rate_map)
        global_maxima = firing_rate_map[global_maxima_bin_idx]

        field_threshold = 0.2*global_maxima

        for local_maximum_idx in local_maxima_bin_idx:
            neighbouring_local_mins = find_neighbouring_minima(firing_rate_map, local_maximum_idx)
            closest_minimum_bin_idx = neighbouring_local_mins[np.argmin(np.abs(neighbouring_local_mins-local_maximum_idx))]
            field_size_in_bins = neighbouring_local_mins[1]-neighbouring_local_mins[0]

            if firing_rate_map[local_maximum_idx] - firing_rate_map[closest_minimum_bin_idx] > field_threshold and field_size_in_bins>5:
                #firing_field.append(neighbouring_local_mins)

                field =  firing_rate_map[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
                field_bins = bin_centres[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
                field_weights = field/np.sum(field)

                field_com = np.sum(field_weights*field_bins)
                firing_field_com.append(field_com)
                firing_field_com_trial_numbers.append(trial_number)
                firing_field_com_trial_types.append(trial_type)

        firing_rate_maps.append(firing_rate_map)

    return firing_field_com, firing_field_com_trial_numbers, firing_field_com_trial_types, firing_rate_maps


def find_neighbouring_minima(firing_rate_map, local_maximum_idx):
    # walk right
    local_min_right = local_maximum_idx
    local_min_right_found = False
    for i in np.arange(local_maximum_idx, len(firing_rate_map)): #local max to end
        if local_min_right_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_right]:
                local_min_right = i
            elif firing_rate_map[i] > firing_rate_map[local_min_right]:
                local_min_right_found = True

    # walk left
    local_min_left = local_maximum_idx
    local_min_left_found = False
    for i in np.arange(0, local_maximum_idx)[::-1]: # local max to start
        if local_min_left_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_left]:
                local_min_left = i
            elif firing_rate_map[i] > firing_rate_map[local_min_left]:
                local_min_left_found = True

    return (local_min_left, local_min_right)


def extract_instantaneous_firing_rate_for_spike(cluster_data, prm):
    firing_times=cluster_data.firing_times/(prm.get_sampling_rate()/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+500, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    smoothened_instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    inds = np.digitize(firing_times, bins)

    ifr = []
    for i in inds:
        ifr.append(smoothened_instantaneous_firing_rate[i-1])

    smoothened_instantaneous_firing_rate_per_spike = np.array(ifr)
    return smoothened_instantaneous_firing_rate_per_spike

def extract_instantaneous_firing_rate_for_spike2(cluster_data, prm):
    firing_times=cluster_data.firing_times/(prm.get_sampling_rate()/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+2000, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    return instantaneous_firing_rate

def process_vr_grid(spike_data, position_data, bin_size, prm):

    fields_com_cluster = []
    fields_com_trial_numbers_cluster = []
    fields_com_trial_types_cluster = []
    firing_rate_maps_cluster = []

    minimum_distance_to_field_in_next_trial =[]
    fields_com_next_trial_type = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        fields_com, field_com_trial_numbers, field_com_trial_types, firing_rate_maps = calculate_grid_field_com(cluster_df, position_data, bin_size, prm)

        if len(firing_rate_maps)==0:
            print("stop here")

        next_trial_type_cluster = []
        minimum_distance_to_field_in_next_trial_cluster=[]

        for i in range(len(fields_com)):
            field = fields_com[i]
            trial_number=field_com_trial_numbers[i]
            trial_type = int(field_com_trial_types[i])

            trial_type_tmp = position_data["trial_type"].to_numpy()
            trial_number_tmp = position_data["trial_number"].to_numpy()

            fields_in_next_trial = np.array(fields_com)[np.array(field_com_trial_numbers) == int(trial_number+1)]
            fields_in_next_trial = fields_in_next_trial[(fields_in_next_trial>50) & (fields_in_next_trial<150)]

            if len(fields_in_next_trial)>0:
                next_trial_type = int(np.unique(trial_type_tmp[trial_number_tmp == int(trial_number+1)])[0])
                minimum_field_difference = min(np.abs(fields_in_next_trial-field))

                minimum_distance_to_field_in_next_trial_cluster.append(minimum_field_difference)
                next_trial_type_cluster.append(next_trial_type)
            else:
                minimum_distance_to_field_in_next_trial_cluster.append(np.nan)
                next_trial_type_cluster.append(np.nan)

        fields_com_cluster.append(fields_com)
        fields_com_trial_numbers_cluster.append(field_com_trial_numbers)
        fields_com_trial_types_cluster.append(field_com_trial_types)
        firing_rate_maps_cluster.append(firing_rate_maps)

        minimum_distance_to_field_in_next_trial.append(minimum_distance_to_field_in_next_trial_cluster)
        fields_com_next_trial_type.append(next_trial_type_cluster)

    spike_data["fields_com"] = fields_com_cluster
    spike_data["fields_com_trial_number"] = fields_com_trial_numbers_cluster
    spike_data["fields_com_trial_type"] = fields_com_trial_types_cluster
    spike_data["firing_rate_maps"] = firing_rate_maps_cluster

    spike_data["minimum_distance_to_field_in_next_trial"] = minimum_distance_to_field_in_next_trial
    spike_data["fields_com_next_trial_type"] = fields_com_next_trial_type

    return spike_data

def calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type):

    cluster_firing_com = np.array(cluster_df["fields_com"].iloc[0])
    cluster_firing_com_trial_types = np.array(cluster_df["fields_com_trial_type"].iloc[0])

    if trial_type == "beaconed":
        n_trials = processed_position_data.beaconed_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 0]
    elif trial_type == "non-beaconed":
        n_trials = processed_position_data.nonbeaconed_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 1]
    elif trial_type == "probe":
        n_trials = processed_position_data.probe_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 2]
    else:
        print("no valid trial type was given")

    if n_trials==0:
        return np.nan
    else:
        return len(firing_com)/n_trials

def process_vr_field_stats(spike_data, processed_position_data, prm):
    n_beaconed_fields_per_trial = []
    n_nonbeaconed_fields_per_trial = []
    n_probe_fields_per_trial = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        n_beaconed_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="beaconed"))
        n_nonbeaconed_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="non-beaconed"))
        n_probe_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="probe"))

    spike_data["n_beaconed_fields_per_trial"] = n_beaconed_fields_per_trial
    spike_data["n_nonbeaconed_fields_per_trial"] = n_nonbeaconed_fields_per_trial
    spike_data["n_probe_fields_per_trial"] = n_probe_fields_per_trial

    return spike_data

def process_vr_field_distances(spike_data, processed_position_data, prm):
    distance_between_fields = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        cluster_firing_com = np.array(cluster_df["fields_com"].iloc[0])
        cluster_firing_com_trial_types = np.array(cluster_df["fields_com_trial_type"].iloc[0])
        cluster_firing_com_trial_numbers = np.array(cluster_df["fields_com_trial_number"].iloc[0])

        distance_covered = (cluster_firing_com_trial_numbers*prm.get_track_length())-prm.get_track_length() #total elapsed distance
        cluster_firing_com = cluster_firing_com+distance_covered

        cluster_firing_com_distance_between = np.diff(cluster_firing_com)
        distance_between_fields.append(cluster_firing_com_distance_between)

    spike_data["distance_between_fields"] = distance_between_fields

    return spike_data


def process_recordings(recording_path_list, bin_size, params):

    for i in range(len(recording_path_list)):
        recording = recording_path_list[i]

        try:

            params = PostSorting.parameters.Parameters()
            params.set_sampling_rate(30000)

            params.set_output_path(recording+"/MountainSort")
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")

            spike_data = process_vr_grid(spike_data, position_data, bin_size, params)
            PostSorting.vr_make_plots.plot_firing_rate_maps_per_trial(spike_data=spike_data, prm=params)
            spike_data = process_vr_field_stats(spike_data, processed_position_data, params)
            spike_data = process_vr_field_distances(spike_data, processed_position_data, params)
            PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed", "non_beaconed", "probe"])
            PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed"])
            PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["non_beaconed"])
            PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["probe"])
            PostSorting.vr_make_plots.plot_field_com_histogram(spike_data=spike_data, prm=params)
            PostSorting.vr_make_plots.plot_inter_field_distance_histogram(spike_data=spike_data, prm=params)

            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            print("successfully processed and saved vr_grid analysis on "+recording)
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)



