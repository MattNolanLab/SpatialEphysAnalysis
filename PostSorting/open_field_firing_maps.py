from joblib import Parallel, delayed
import os
import multiprocessing
import matplotlib.pylab as plt
import pandas as pd
from numba import jit
import numpy as np
import math
import time

try_parallel_first = True

def get_dwell(spatial_data, prm):
    min_dwell_distance_cm = 5  # from point to determine min dwell time
    min_dwell_distance_pixels = min_dwell_distance_cm / 100 * prm.get_pixel_ratio()

    dt_position_ms = spatial_data.synced_time.diff().mean()*1000  # average sampling interval in position data (ms)
    min_dwell_time_ms = 3 * dt_position_ms  # this is about 100 ms
    min_dwell = round(min_dwell_time_ms/dt_position_ms)
    return min_dwell, min_dwell_distance_pixels


def get_bin_size(prm):
    bin_size_cm = 2.5
    bin_size_pixels = bin_size_cm / 100 * prm.get_pixel_ratio()
    return bin_size_pixels


def get_number_of_bins(spatial_data, prm):
    bin_size_pixels = get_bin_size(prm)
    length_of_arena_x = spatial_data.position_x_pixels[~np.isnan(spatial_data.position_x_pixels)].max()
    length_of_arena_y = spatial_data.position_y_pixels[~np.isnan(spatial_data.position_y_pixels)].max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_pixels)
    number_of_bins_y = math.ceil(length_of_arena_y / bin_size_pixels)
    return number_of_bins_x, number_of_bins_y


@jit
def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny


def calculate_firing_rate_for_cluster_parallel(cluster_id, smooth, firing_data_spatial, positions_x, positions_y, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    print('Started another cluster')
    print(cluster_id)
    #cluster_index = firing_data_spatial.cluster_id.values[cluster] - 1

    cluster_firing_data_spatial = firing_data_spatial[firing_data_spatial.cluster_id == cluster_id]
    cluster_firings = pd.DataFrame({'position_x': cluster_firing_data_spatial.position_x_pixels.iloc[0], 'position_y': cluster_firing_data_spatial.position_y_pixels.iloc[0]})

    #cluster_firings = pd.DataFrame({'position_x': firing_data_spatial.position_x_pixels[cluster_index], 'position_y': firing_data_spatial.position_y_pixels[cluster_index]})


    spike_positions_x = cluster_firings.position_x.values
    spike_positions_y = cluster_firings.position_y.values

    spike_positions_y = spike_positions_y[~np.isnan(spike_positions_y)]
    spike_positions_x = spike_positions_x[~np.isnan(spike_positions_x)]
    positions_y = positions_y[~np.isnan(positions_y)]
    positions_x = positions_x[~np.isnan(positions_x)]

    x = np.linspace((bin_size_pixels/2), (bin_size_pixels*number_of_bins_x)-(bin_size_pixels/2), number_of_bins_x)
    y = np.linspace((bin_size_pixels/2), (bin_size_pixels*number_of_bins_y)-(bin_size_pixels/2), number_of_bins_y)

    xv, yv = np.meshgrid(x, y)

    xv_spikes = np.repeat(xv[:, :, np.newaxis], len(spike_positions_x), axis=2)
    yv_spikes = np.repeat(yv[:, :, np.newaxis], len(spike_positions_y), axis=2)
    xv_spikes = xv_spikes - spike_positions_x
    yv_spikes = yv_spikes - spike_positions_y
    xv_spikes = np.power(xv_spikes, 2)
    yv_spikes = np.power(yv_spikes, 2)

    xy_spikes = xv_spikes+yv_spikes
    xy_spikes = np.sqrt(xy_spikes)
    xy_spikes = xy_spikes/smooth
    xy_spikes = gaussian_kernel(xy_spikes)
    xy_spikes = np.sum(xy_spikes, axis=2)
    xy_spikes[np.isnan(xy_spikes)] = 0

    xv_locs = np.repeat(xv[:, :, np.newaxis], len(positions_x), axis=2)
    yv_locs = np.repeat(yv[:, :, np.newaxis], len(positions_y), axis=2)
    xv_locs = xv_locs - positions_x
    yv_locs = yv_locs - positions_y
    xv_locs = np.power(xv_locs, 2)
    yv_locs = np.power(yv_locs, 2)

    xy_locs = xv_locs+yv_locs
    xy_locs = np.sqrt(xy_locs)

    occupancies = np.sum((xy_locs<min_dwell_distance_pixels).astype(int), axis=2)
    occupancies[occupancies < min_dwell] = 0
    occupancies[occupancies!=0] = 1

    xy_locs = xy_locs/smooth
    xy_locs = gaussian_kernel(xy_locs)
    xy_locs = np.sum(xy_locs, axis=2)
    xy_locs[np.isnan(xy_locs)] = 0
    xy_locs = xy_locs*(dt_position_ms/1000)

    firing_rate_map = np.divide(xy_spikes, xy_locs)
    firing_rate_map = firing_rate_map*occupancies # occupancies is a mask

    return np.transpose(firing_rate_map)

def calculate_firing_rate_for_cluster_parallel_old(cluster, smooth, firing_data_spatial, positions_x, positions_y, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    print('Started another cluster')
    print(cluster)
    cluster_index = firing_data_spatial.cluster_id.values[cluster] - 1
    cluster_firings = pd.DataFrame({'position_x': firing_data_spatial.position_x_pixels[cluster_index], 'position_y': firing_data_spatial.position_y_pixels[cluster_index]})
    spike_positions_x = cluster_firings.position_x.values
    spike_positions_y = cluster_firings.position_y.values

    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2) + np.power(py - spike_positions_y, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2) + np.power((py - positions_y), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                firing_rate_map[x, y] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))

            else:
                firing_rate_map[x, y] = 0
    #firing_rate_map = np.rot90(firing_rate_map)

    return firing_rate_map

def get_spike_heatmap_parallel(spatial_data, firing_data_spatial, prm):
    print('I will calculate firing rate maps now.')
    dt_position_ms = spatial_data.synced_time.diff().mean()*1000
    min_dwell, min_dwell_distance_pixels = get_dwell(spatial_data, prm)
    smooth = 5 / 100 * prm.get_pixel_ratio()
    bin_size_pixels = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)

    clusters = range(len(firing_data_spatial))
    clusters = firing_data_spatial.cluster_id

    num_cores = int(os.environ['HEATMAP_CONCURRENCY']) if os.environ.get('HEATMAP_CONCURRENCY') else multiprocessing.cpu_count()
    time_start = time.time()
    if try_parallel_first:
        try:
            firing_rate_maps = Parallel(n_jobs=num_cores)(delayed(calculate_firing_rate_for_cluster_parallel)(cluster, smooth, firing_data_spatial, spatial_data.position_x_pixels.values, spatial_data.position_y_pixels.values, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms) for cluster in clusters)
        except Exception as ex:
            print("calculating rate map failed using parallel, attempting one by one")
            firing_rate_maps = []
            for cluster in clusters:
                firing_rate_maps.append(calculate_firing_rate_for_cluster_parallel(cluster, smooth, firing_data_spatial,
                                                                                   spatial_data.position_x_pixels.values,
                                                                                   spatial_data.position_y_pixels.values,
                                                                                   number_of_bins_x, number_of_bins_y,
                                                                                   bin_size_pixels, min_dwell,
                                                                                   min_dwell_distance_pixels, dt_position_ms))
    else:
        firing_rate_maps = Parallel(n_jobs=num_cores)(delayed(calculate_firing_rate_for_cluster_parallel)(cluster, smooth, firing_data_spatial, spatial_data.position_x_pixels.values, spatial_data.position_y_pixels.values, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms) for cluster in clusters)

    time_end = time.time()
    print('Making the rate maps took ', time_end-time_start, " seconds")
    firing_data_spatial['firing_maps'] = firing_rate_maps

    return firing_data_spatial


def get_position_heatmap_fixed_bins(spatial_data, number_of_bins_x, number_of_bins_y, bin_size_cm, min_dwell_distance_cm, min_dwell):
    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_cm + (bin_size_cm / 2)
            py = y * bin_size_cm + (bin_size_cm / 2)

            occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x_pixels.values), 2) + np.power((py - spatial_data.position_y_pixels.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_cm)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = None
    return position_heat_map


def get_position_heatmap(spatial_data, prm):
    min_dwell, min_dwell_distance_cm = get_dwell(spatial_data, prm)
    bin_size_cm = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)

    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))

    # find value for each bin for heatmap
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_cm + (bin_size_cm / 2)
            py = y * bin_size_cm + (bin_size_cm / 2)

            occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x_pixels.values), 2) + np.power((py - spatial_data.position_y_pixels.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_cm)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = None
    return position_heat_map


# this is the firing rate in the bin with the highest rate
def find_maximum_firing_rate(spatial_firing):
    max_firing_rates = []
    #for cluster in range(len(spatial_firing)):
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        #cluster = spatial_firing.cluster_id.values[cluster] - 1
        #firing_rate_map = spatial_firing.firing_maps[cluster]
        firing_rate_map = cluster_df.firing_maps.iloc[0]
        max_firing_rate = np.max(firing_rate_map.flatten())
        max_firing_rates.append(max_firing_rate)
    spatial_firing['max_firing_rate'] = max_firing_rates
    return spatial_firing


def make_firing_field_maps(spatial_data, firing_data_spatial, prm):
    position_heat_map = get_position_heatmap(spatial_data, prm)
    firing_data_spatial = get_spike_heatmap_parallel(spatial_data, firing_data_spatial, prm)
    #position_heat_map = np.rot90(position_heat_map)  # to rotate map to be like matlab plots
    firing_data_spatial = find_maximum_firing_rate(firing_data_spatial)
    return position_heat_map, firing_data_spatial