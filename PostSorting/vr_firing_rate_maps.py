import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import PostSorting.vr_sync_spatial_data
import os
from scipy import stats

def get_total_bin_times(binned_times_collumn):
    # this function adds all the binned times per trial to give the total
    # time spent in a location bin for a given processed_position_data-like dataframe
    total_bin_times = np.zeros(len(binned_times_collumn.iloc[0]))
    for i in range(len(binned_times_collumn)):
        dwell_time = binned_times_collumn.iloc[i]
        assert not np.any(np.isnan(dwell_time)), f'invalid dwell time encountered {dwell_time}' #make sure the time is valid
        total_bin_times += dwell_time
    return total_bin_times

def calculate_rate_map_sem(spike_locations, spike_trial_numbers, processed_position_data, bins):

    rate_map_per_trial = []
    for trial_number in np.unique(processed_position_data["trial_number"]):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        trial_spike_locations = spike_locations[spike_trial_numbers==trial_number]
        trial_binned_time = trial_processed_position_data["times_binned"].iloc[0]
        spike_bin_counts = np.histogram(trial_spike_locations, bins)[0]
        normalised_rate_map = spike_bin_counts/trial_binned_time

        rate_map_per_trial.append(normalised_rate_map.tolist())

    rate_map_per_trial = np.array(rate_map_per_trial)
    rate_map_sem = stats.sem(rate_map_per_trial, axis=0, nan_policy="omit")

    return np.array(rate_map_sem)

def make_firing_field_maps(spike_data, processed_position_data, bin_size_cm, track_length):

    beaconed_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 1]
    probe_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 2]

    bins = np.arange(0, track_length, bin_size_cm)

    beaconed_firing_rate_map = []
    non_beaconed_firing_rate_map = []
    probe_firing_rate_map = []

    beaconed_firing_rate_map_sem = []
    non_beaconed_firing_rate_map_sem = []
    probe_firing_rate_map_sem = []

    print('I am calculating the average firing rate ...')
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        trial_numbers = np.array(cluster_spike_data["trial_number"].tolist()[0])
        trial_types = np.array(cluster_spike_data["trial_type"].tolist()[0])
        x_locations_cm = np.array(cluster_spike_data["x_position_cm"].tolist()[0])

        if len(beaconed_processed_position_data)>0:
            cluster_trial_x_locations = x_locations_cm[trial_types == 0]
            cluster_trial_numbers = trial_numbers[trial_types == 0]
            beaconed_bin_counts = np.histogram(cluster_trial_x_locations, bins)[0]
            binned_times = get_total_bin_times(beaconed_processed_position_data["times_binned"])
            normalised_rate_map = beaconed_bin_counts/binned_times
            beaconed_firing_rate_map.append(normalised_rate_map.tolist())

            # calculate standard error
            beaconed_firing_rate_map_sem.append(calculate_rate_map_sem(spike_locations=cluster_trial_x_locations,
                                                                       spike_trial_numbers=cluster_trial_numbers,
                                                                       processed_position_data=beaconed_processed_position_data,
                                                                       bins=bins))

        if len(non_beaconed_processed_position_data)>0:
            cluster_trial_x_locations = x_locations_cm[trial_types == 1]
            cluster_trial_numbers = trial_numbers[trial_types == 1]
            non_beaconed_bin_counts = np.histogram(cluster_trial_x_locations, bins)[0]
            binned_times = get_total_bin_times(non_beaconed_processed_position_data["times_binned"])
            normalised_rate_map = non_beaconed_bin_counts/binned_times
            non_beaconed_firing_rate_map.append(normalised_rate_map.tolist())

            # calculate standard error
            non_beaconed_firing_rate_map_sem.append(calculate_rate_map_sem(spike_locations=cluster_trial_x_locations,
                                                                           spike_trial_numbers=cluster_trial_numbers,
                                                                           processed_position_data=non_beaconed_processed_position_data,
                                                                           bins=bins))

        if len(probe_processed_position_data)>0:
            cluster_trial_x_locations = x_locations_cm[trial_types == 2]
            cluster_trial_numbers = trial_numbers[trial_types == 2]
            probe_bin_counts = np.histogram(cluster_trial_x_locations, bins)[0]
            binned_times = get_total_bin_times(probe_processed_position_data["times_binned"])
            normalised_rate_map = probe_bin_counts/binned_times
            probe_firing_rate_map.append(normalised_rate_map.tolist())

            # calculate standard error
            probe_firing_rate_map_sem.append(calculate_rate_map_sem(spike_locations=cluster_trial_x_locations,
                                                                    spike_trial_numbers=cluster_trial_numbers,
                                                                    processed_position_data=probe_processed_position_data,
                                                                    bins=bins))

        else:
            probe_firing_rate_map.append([])
            probe_firing_rate_map_sem.append([])
            # pass an empty list when probe trials are not present

    spike_data["beaconed_firing_rate_map"] = beaconed_firing_rate_map
    spike_data["non_beaconed_firing_rate_map"] = non_beaconed_firing_rate_map
    spike_data["probe_firing_rate_map"] = probe_firing_rate_map

    spike_data["beaconed_firing_rate_map_sem"] = beaconed_firing_rate_map_sem
    spike_data["non_beaconed_firing_rate_map_sem"] = non_beaconed_firing_rate_map_sem
    spike_data["probe_firing_rate_map_sem"] = probe_firing_rate_map_sem

    print('-------------------------------------------------------------')
    print('firing field maps processed for all trials')
    print('-------------------------------------------------------------')
    return spike_data