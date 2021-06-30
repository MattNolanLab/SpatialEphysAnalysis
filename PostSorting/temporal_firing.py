import numpy as np
import settings


def add_temporal_firing_properties_to_df(spatial_firing, total_length_seconds):
    # calculate number of spikes and mean firing rate for each cluster and add to spatial firing df
    total_number_of_spikes_per_cluster = []
    mean_firing_rates = []
    for cluster, cluster_id in enumerate(spatial_firing.cluster_id):
        firing_times = np.asarray(spatial_firing[spatial_firing.cluster_id == cluster_id].firing_times)[0]
        total_number_of_spikes = len(firing_times)
        mean_firing_rate = total_number_of_spikes / total_length_seconds  # this does not include opto
        total_number_of_spikes_per_cluster.append(total_number_of_spikes)
        mean_firing_rates.append(mean_firing_rate)
    spatial_firing['number_of_spikes'] = total_number_of_spikes_per_cluster
    spatial_firing['mean_firing_rate'] = mean_firing_rates
    spatial_firing['recording_length_seconds'] = total_length_seconds
    return spatial_firing





