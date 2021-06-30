import pandas as pd
import numpy as np
from numpy import inf

def extract_instant_rates(spike_data, cluster_index):
    speed = np.array(spike_data.loc[cluster_index].instant_rates[0])
    location = np.array(spike_data.loc[cluster_index].instant_rates[1])
    firing_rate = np.array(spike_data.loc[cluster_index].instant_rates[2])
    return speed, location, firing_rate


def extract_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.iloc[cluster_index].spike_rate_on_trials[0], 'trial_number' :  spike_data.loc[cluster_index].spike_rate_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].spike_rate_on_trials[2]})
    return cluster_firings


def extract_smoothed_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[0], 'trial_number' :  spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[1], 'trial_type' :  spike_data.iloc[cluster_index].spike_rate_on_trials_smoothed[2]})
    return cluster_firings


def extract_gc_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.iloc[cluster_index].firing_maps, 'trial_number' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[1], 'trial_type' :  spike_data.loc[cluster_index].spike_rate_on_trials_smoothed[2]})
    return cluster_firings


def extract_firing_num_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.iloc[cluster_index].spike_num_on_trials[0], 'trial_number' :  spike_data.loc[cluster_index].spike_num_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].spike_num_on_trials[2]})
    return cluster_firings


def extract_shuffled_firing_num_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[0], 'trial_number' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[2], 'bins' :  spike_data.loc[cluster_index].shuffled_spike_num_on_trials[3]})
    return cluster_firings


def extract_shuffled_firing_rate_data(spike_data, cluster_index):
    cluster_firings = pd.DataFrame({ 'firing_rate' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[0], 'trial_number' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[1], 'trial_type' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[2], 'bins' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[3]})
    return cluster_firings


def extract_average_speed_data(spike_data, cluster_index):
    cluster_speed = pd.DataFrame({ 'speed_b' :  spike_data.loc[cluster_index].shuffled_spike_rate_on_trials[0], 'speed_nb' :  spike_data.loc[cluster_index].average_speed[1], 'speed_p' :  spike_data.loc[cluster_index].average_speed[2], 'bins' :  spike_data.loc[cluster_index].average_speed[3]})
    return cluster_speed


def split_firing_data_by_trial_type(cluster_firings):
    beaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==0)
    nbeaconed_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==1)
    probe_cluster_firings = cluster_firings.where(cluster_firings["trial_type"] ==2)
    return beaconed_cluster_firings, nbeaconed_cluster_firings, probe_cluster_firings


def reshape_and_average_over_trials(beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings, max_trial_number, prm):
    beaconed_cluster_firings[beaconed_cluster_firings == inf] = 0
    nonbeaconed_cluster_firings[nonbeaconed_cluster_firings == inf] = 0
    probe_cluster_firings[probe_cluster_firings == inf] = 0

    beaconed_reshaped_hist = np.reshape(beaconed_cluster_firings, (int(beaconed_cluster_firings.size/prm.track_length), int(prm.track_length)))
    nonbeaconed_reshaped_hist = np.reshape(nonbeaconed_cluster_firings, (int(nonbeaconed_cluster_firings.size/prm.track_length), int(prm.track_length)))
    probe_reshaped_hist = np.reshape(probe_cluster_firings, (int(probe_cluster_firings.size/prm.track_length), int(prm.track_length)))
    average_beaconed_spike_rate = np.nanmean(beaconed_reshaped_hist, axis=0)
    average_nonbeaconed_spike_rate = np.nanmean(nonbeaconed_reshaped_hist, axis=0)
    average_probe_spike_rate = np.nanmean(probe_reshaped_hist, axis=0)
    average_beaconed_spike_rate = np.nan_to_num(average_beaconed_spike_rate, copy=True)
    average_nonbeaconed_spike_rate = np.nan_to_num(average_nonbeaconed_spike_rate, copy=True)
    average_probe_spike_rate = np.nan_to_num(average_probe_spike_rate, copy=True)

    average_beaconed_spike_rate[average_beaconed_spike_rate == inf] = 0
    average_nonbeaconed_spike_rate[average_nonbeaconed_spike_rate == inf] = 0
    average_probe_spike_rate[average_probe_spike_rate == inf] = 0

    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16)


def extract_average_shuffled_firing_rate_data(spike_data, cluster_index):
    cluster_firings = extract_shuffled_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return np.array(average_beaconed_spike_rate, dtype=np.float16), np.array(average_nonbeaconed_spike_rate, dtype=np.float16), np.array(average_probe_spike_rate, dtype=np.float16)


def extract_average_firing_rate_data(spike_data, cluster_index):
    cluster_firings = extract_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"], dtype=np.float16), np.array(nonbeaconed_cluster_firings["firing_rate"], dtype=np.float16), np.array(probe_cluster_firings["firing_rate"], dtype=np.float16), max(cluster_firings["trial_number"]))
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate


def extract_smoothed_average_firing_rate_data(spike_data, cluster_index, prm):
    cluster_firings = extract_smoothed_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"], dtype=np.float16), np.array(nonbeaconed_cluster_firings["firing_rate"], dtype=np.float16), np.array(probe_cluster_firings["firing_rate"], dtype=np.float16), max(cluster_firings["trial_number"]), prm)
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate


def extract_gc_average_firing_rate_data(spike_data, cluster_index):
    cluster_firings = extract_gc_firing_rate_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate


def extract_average_firing_num_data(spike_data, cluster_index):
    cluster_firings = extract_firing_num_data(spike_data, cluster_index)
    beaconed_cluster_firings, nonbeaconed_cluster_firings, probe_cluster_firings = split_firing_data_by_trial_type(cluster_firings)
    average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate = reshape_and_average_over_trials(np.array(beaconed_cluster_firings["firing_rate"]), np.array(nonbeaconed_cluster_firings["firing_rate"]), np.array(probe_cluster_firings["firing_rate"]), max(cluster_firings["trial_number"]))
    return average_beaconed_spike_rate, average_nonbeaconed_spike_rate, average_probe_spike_rate
