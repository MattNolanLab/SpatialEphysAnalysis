import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
from scipy import stats
import PostSorting.vr_speed_analysis
import plot_utility

def calculate_total_trial_numbers(raw_position_data,processed_position_data):
    print('calculating total trial numbers for trial types')
    trial_numbers = np.array(raw_position_data['trial_number'])
    trial_type = np.array(raw_position_data['trial_type'])
    trial_data=np.transpose(np.vstack((trial_numbers, trial_type)))
    beaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]>0),0)
    unique_beaconed_trials = np.unique(beaconed_trials[:,0])
    nonbeaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]!=1),0)
    unique_nonbeaconed_trials = np.unique(nonbeaconed_trials[1:,0])
    probe_trials = np.delete(trial_data, np.where(trial_data[:,1]!=2),0)
    unique_probe_trials = np.unique(probe_trials[1:,0])

    processed_position_data.at[0,'beaconed_total_trial_number'] = len(unique_beaconed_trials)
    processed_position_data.at[0,'nonbeaconed_total_trial_number'] = len(unique_nonbeaconed_trials)
    processed_position_data.at[0,'probe_total_trial_number'] = len(unique_probe_trials)
    return processed_position_data

def trial_average_speed(processed_position_data):
    # split binned speed data by trial type
    beaconed = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed = processed_position_data[processed_position_data["trial_type"] == 1]
    probe = processed_position_data[processed_position_data["trial_type"] == 2]

    if len(beaconed)>0:
        beaconed_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(beaconed["speeds_binned"])
        trial_averaged_beaconed_speeds = np.nanmean(beaconed_speeds, axis=0)
    else:
        trial_averaged_beaconed_speeds = np.array([])

    if len(non_beaconed)>0:
        non_beaconed_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(non_beaconed["speeds_binned"])
        trial_averaged_non_beaconed_speeds = np.nanmean(non_beaconed_speeds, axis=0)
    else:
        trial_averaged_non_beaconed_speeds = np.array([])

    if len(probe)>0:
        probe_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(probe["speeds_binned"])
        trial_averaged_probe_speeds = np.nanmean(probe_speeds, axis=0)
    else:
        trial_averaged_probe_speeds = np.array([])

    return trial_averaged_beaconed_speeds, trial_averaged_non_beaconed_speeds, trial_averaged_probe_speeds


def process_position(raw_position_data, stop_threshold,track_length):
    processed_position_data = pd.DataFrame() # make dataframe for processed position data
    processed_position_data = PostSorting.vr_speed_analysis.process_speed(raw_position_data, processed_position_data, track_length)
    processed_position_data = PostSorting.vr_time_analysis.process_time(raw_position_data, processed_position_data,track_length)
    processed_position_data = PostSorting.vr_stop_analysis.process_stops(processed_position_data, stop_threshold)
    gc.collect()

    processed_position_data["new_trial_indices"] = raw_position_data["new_trial_indices"].dropna()

    print('-------------------------------------------------------------')
    print('position data processed')
    print('-------------------------------------------------------------')
    return processed_position_data



