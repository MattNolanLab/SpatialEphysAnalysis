import pandas as pd


def calculate_corresponding_indices(spike_data, spatial_data, sampling_rate_ephys=30000):
    # this is needed when multiple recordings are stitched together for sorting
    firing_times = spike_data.firing_times
    avg_sampling_rate_bonsai = float(1 / spatial_data['synced_time'].diff().mean())
    sampling_rate_rate = sampling_rate_ephys / avg_sampling_rate_bonsai
    spike_data['bonsai_indices'] = firing_times / sampling_rate_rate
    return spike_data


def find_firing_location_indices(spike_data, spatial_data):
    spike_data = calculate_corresponding_indices(spike_data, spatial_data)
    spatial_firing = pd.DataFrame(columns=['position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'speed'])

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        bonsai_indices_cluster = cluster_df.bonsai_indices.iloc[0]
        bonsai_indices_cluster_round = bonsai_indices_cluster.round(0)
        spatial_firing = spatial_firing.append({
            "position_x": list(spatial_data.position_x[bonsai_indices_cluster_round]),
            "position_x_pixels": list(spatial_data.position_x_pixels[bonsai_indices_cluster_round]),
            "position_y":  list(spatial_data.position_y[bonsai_indices_cluster_round]),
            "position_y_pixels":  list(spatial_data.position_y_pixels[bonsai_indices_cluster_round]),
            "hd": list(spatial_data.hd[bonsai_indices_cluster_round]),
            "speed": list(spatial_data.speed[bonsai_indices_cluster_round])
        }, ignore_index=True)
    spike_data['position_x'] = spatial_firing.position_x.values
    spike_data['position_x_pixels'] = spatial_firing.position_x_pixels.values
    spike_data['position_y'] = spatial_firing.position_y.values
    spike_data['position_y_pixels'] = spatial_firing.position_y_pixels.values
    spike_data['hd'] = spatial_firing.hd.values
    spike_data['speed'] = spatial_firing.speed.values
    spike_data = spike_data.drop(['bonsai_indices'], axis=1)
    return spike_data


def process_spatial_firing(spike_data, spatial_data):
    """
    :param paired_order: number of recording in series of recordings sorted together
    :param stitch_point: list of points where recordings sorted together were combined
    :param spike_data: data frame containing firing times where each row is a neuron
    :param spatial_data: data frame containing position of animal (x, y, hd, time)
    :return: combined data frame containing firing times and corresponding positions as lists
    firing_times = [,,,,] x = [,,,,] y = [,,,,] ... in each row for each cell
    """
    if 'position_x' in spike_data:
        return spike_data
    spatial_spike_data = find_firing_location_indices(spike_data, spatial_data)
    return spatial_spike_data
