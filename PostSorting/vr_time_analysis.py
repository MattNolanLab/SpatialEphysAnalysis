import numpy as np
import pandas as pd
import PostSorting.parameters
import matplotlib.pyplot as plt
import settings

def calculate_binned_time(raw_position_data,processed_position_data, track_length):
    bin_size_cm = settings.vr_bin_size_cm

    times_binned = []
    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number])
        trial_times = np.array(raw_position_data['dwell_time_ms'][np.array(raw_position_data['trial_number']) == trial_number])

        bins = np.arange(0, track_length, bin_size_cm)
        bin_times = np.histogram(trial_x_position_cm, bins, weights=trial_times)[0]

        times_binned.append(bin_times)

    processed_position_data['times_binned'] = times_binned
    return processed_position_data

def process_time(raw_position_data,processed_position_data, track_length):
    processed_position_data = calculate_binned_time(raw_position_data,processed_position_data, track_length)
    return processed_position_data

