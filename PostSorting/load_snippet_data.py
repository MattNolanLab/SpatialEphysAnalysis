import os
import mdaio
import numpy as np
import PreClustering.dead_channels
import matplotlib.pylab as plt


def extract_random_snippets(filtered_data, firing_times, tetrode, number_of_snippets, dead_channels):
    if len(dead_channels) != 0:
        for dead_ch in range(len(dead_channels[0])):
            to_insert = np.zeros(len(filtered_data[0]))
            filtered_data = np.insert(filtered_data, int(dead_channels[0][dead_ch]) - 1, to_insert, 0)

    if len(firing_times)<50:
        random_indices = np.arange(0, len(firing_times))
    else:
        random_indices = np.ceil(np.random.uniform(16, len(firing_times)-16, number_of_snippets)).astype(int)
    snippets = np.zeros((4, 30, number_of_snippets))

    channels = [(tetrode-1)*4, (tetrode-1)*4 + 1, (tetrode-1)*4 + 2, (tetrode-1)*4 + 3]

    for index, event in enumerate(random_indices):
        snippets_indices = np.arange(firing_times[event]-10, firing_times[event]+20, 1).astype(int)
        snippets[:, :, index] = filtered_data[channels[0]:channels[3]+1, snippets_indices]
    # plt.plot(snippets[3,:,:]) # example ch plot
    return snippets

def extract_all_snippets(filtered_data, firing_times, tetrode, dead_channels):
    if len(dead_channels) != 0:
        for dead_ch in range(len(dead_channels[0])):
            to_insert = np.zeros(len(filtered_data[0]))
            filtered_data = np.insert(filtered_data, int(dead_channels[0][dead_ch]) - 1, to_insert, 0)

    if len(firing_times)<50:
        all_indices = np.arange(0, len(firing_times))
    else:
        all_indices = np.arange(16, len(firing_times)-16)
    snippets = np.zeros((4, 30, len(all_indices)))

    channels = [(tetrode-1)*4, (tetrode-1)*4 + 1, (tetrode-1)*4 + 2, (tetrode-1)*4 + 3]

    for index, event in enumerate(all_indices):
        snippets_indices = np.arange(firing_times[event]-10, firing_times[event]+20, 1).astype(int)
        snippets[:, :, index] = filtered_data[channels[0]:channels[3]+1, snippets_indices]
    # plt.plot(snippets[3,:,:]) # example ch plot
    return snippets


def get_snippets(firing_data, file_path, sorter_name, dead_channels, random_snippets=True):
    if 'random_snippets' in firing_data:
        return firing_data
    print('I will get some random snippets now for each cluster.')

    filtered_data_path = file_path + '/Electrophysiology/' + sorter_name + '/filt.mda'

    snippets_all_clusters = []
    if os.path.exists(filtered_data_path):
        filtered_data = mdaio.readmda(filtered_data_path)
        for cluster, cluster_id in enumerate(firing_data.cluster_id):
            tetrode = np.asarray(firing_data[firing_data.cluster_id == cluster_id].tetrode)[0]
            firing_times = np.asarray(firing_data[firing_data.cluster_id == cluster_id].firing_times)[0]
            if random_snippets is True:
                snippets = extract_random_snippets(filtered_data, firing_times, tetrode, 50, dead_channels)
            else:
                snippets = extract_all_snippets(filtered_data, firing_times, tetrode, dead_channels)
            snippets_all_clusters.append(snippets)

    if random_snippets is True:
        firing_data['random_snippets'] = snippets_all_clusters
    else:
        firing_data['all_snippets'] = snippets_all_clusters
    #plt.plot(firing_data.random_snippets[4][3,:,:])
    return firing_data