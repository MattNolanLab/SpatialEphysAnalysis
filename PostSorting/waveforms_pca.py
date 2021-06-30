import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.toolkit as st
import os
import spikeinterface.extractors as se
import OpenEphys

unique_colors = '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', \
                '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', \
                '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', \
                '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000'

def spatial_firing2label(spatial_firing):
    times = []
    labels = []
    for cluster_id in np.unique(spatial_firing["cluster_id"]):
        cluster_spatial_firing = spatial_firing[(spatial_firing["cluster_id"] == cluster_id)]
        cluster_times = list(cluster_spatial_firing["firing_times"].iloc[0])
        cluster_labels = list(cluster_id*np.ones(len(cluster_times)))

        times.extend(cluster_times)
        labels.extend(cluster_labels)
    return np.array(times), np.array(labels)

def tetrode2letter(tetrode_int):
    if tetrode_int == 1:
        return "A"
    elif tetrode_int == 2:
        return "B"
    elif tetrode_int == 3:
        return "C"
    elif tetrode_int == 4:
        return "D"

def load_OpenEphysRecording(folder, data_file_prefix, num_tetrodes):
    signal = []
    for i in range(num_tetrodes*4):
        fname = folder+'/'+data_file_prefix+str(i+1)+'.continuous'
        x = OpenEphys.loadContinuousFast(fname)['data']
        if i==0:
            #preallocate array on first run
            signal = np.zeros((num_tetrodes*4,x.shape[0]))
        signal[i,:] = x
    return signal

def getDeadChannel(deadChannelFile):
    with open(deadChannelFile,'r') as f:
        deadChannels = [int(s) for s in f.readlines()]
    return deadChannels

def plot_pca_waveforms(spatial_firing, recording, output_folder, remove_outliers=False, sampling_rate=30000):
    fig_combos = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    channel_combos =  [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    signal = load_OpenEphysRecording(recording, data_file_prefix='100_CH', num_tetrodes=4)
    dead_channel_path = recording +'/dead_channels.txt'
    bad_channel = getDeadChannel(dead_channel_path)
    tetrode_geom = '/home/ubuntu/to_sort/sorting_files/geom_all_tetrodes_original.csv'
    geom = pd.read_csv(tetrode_geom,header=None).values
    recording = se.NumpyRecordingExtractor(signal, sampling_rate, geom)
    recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel)
    recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = st.preprocessing.whiten(recording)
    recording = se.CacheRecordingExtractor(recording)

    for tetrode in np.unique(spatial_firing["tetrode"]):
        tetrode_spatial_firing = spatial_firing[(spatial_firing["tetrode"] == tetrode)]
        # reconstruct a sorting extractor
        times, labels = spatial_firing2label(tetrode_spatial_firing)
        sorting = se.NumpySortingExtractor()
        sorting.set_times_labels(times=times, labels=labels)
        sorting.set_sampling_frequency(sampling_frequency=sampling_rate)

        channel_list = [(tetrode-1)*4+0, (tetrode-1)*4+1, (tetrode-1)*4+2, (tetrode-1)*4+3]
        unit_list = list(tetrode_spatial_firing["cluster_id"])

        pca_scores = st.postprocessing.compute_unit_pca_scores(recording, sorting, n_comp=3,
                                                               verbose=True, by_electrode=True,
                                                               unit_ids=unit_list, channel_ids=channel_list,
                                                               ms_before=0.5, ms_after=0.5,
                                                               save_property_or_features=False)

        # we want a plot per tetrode
        fig, axs = plt.subplots(2, 3, gridspec_kw = {'wspace':0, 'hspace':0})
        plt.rcParams['axes.labelsize'] = 15
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        for i in range(len(fig_combos)):
            channel_combo = channel_combos[i]
            fig_combo = fig_combos[i]
            axs[fig_combo[0], fig_combo[1]].set_yticklabels([])
            axs[fig_combo[0], fig_combo[1]].set_xticklabels([])
            axs[fig_combo[0], fig_combo[1]].set_yticks([])
            axs[fig_combo[0], fig_combo[1]].set_xticks([])
            axs[fig_combo[0], fig_combo[1]].text(0.05, 0.9, tetrode2letter(tetrode)+str(channel_combo[0]+1),
                                                 fontsize=15, horizontalalignment='left', verticalalignment='center',
                                                 transform=axs[fig_combo[0], fig_combo[1]].transAxes)
            axs[fig_combo[0], fig_combo[1]].text(0.95, 0.1, tetrode2letter(tetrode)+str(channel_combo[1]+1),
                                                 fontsize=15, horizontalalignment='right', verticalalignment='center',
                                                 transform=axs[fig_combo[0], fig_combo[1]].transAxes)

            for i in range(len(pca_scores)):
                pc1 = pca_scores[i][:, channel_combo[0], 0]
                pc2 = pca_scores[i][:, channel_combo[1], 1]
                if remove_outliers:
                    pc1, pc2 = reject_outliers(pc1, pc2)

                axs[fig_combo[0], fig_combo[1]].plot(pc1, pc2, '.', label=str(unit_list[i]),
                                                     alpha=0.3, color=unique_colors[i], markersize=1)

        leg = axs[fig_combos[-1][0], fig_combos[-1][1]].legend(bbox_to_anchor=(1.7, 2.04), title="Cluster IDs")
        for legend_handle in leg.legendHandles:
            legend_handle._legmarker.set_markersize(10)
            legend_handle._legmarker.set_alpha(1)

        axs[0,0].set_ylabel('PC 2', fontsize=15)
        axs[1,0].set_xlabel('PC 1', fontsize=15)
        axs[1,0].set_ylabel('PC 2', fontsize=15)
        axs[1,1].set_xlabel('PC 1', fontsize=15)
        axs[1,2].set_xlabel('PC 1', fontsize=15)
        fig.subplots_adjust(right=0.8)
        plt.savefig(output_folder+"pca_tetrode"+str(tetrode)+".png", dpi=300)
        plt.show()
        plt.close()
    return

def reject_outliers(pc1, pc2):
    # remove and mask based on pc1
    filtered_pc1_by_pc1 = pc1[~is_outlier(pc1)]
    filtered_pc2_by_pc1 = pc2[~is_outlier(pc1)]
    # remove and mask based on pc2
    filtered_pc1_by_both = filtered_pc1_by_pc1[~is_outlier(filtered_pc2_by_pc1)]
    filtered_pc2_by_both = filtered_pc2_by_pc1[~is_outlier(filtered_pc2_by_pc1)]

    return filtered_pc1_by_both, filtered_pc2_by_both

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def create_phy(recording, spatial_firing, output_folder, sampling_rate=30000):
    signal = load_OpenEphysRecording(recording, data_file_prefix='100_CH', num_tetrodes=4)
    dead_channel_path = recording +'/dead_channels.txt'
    bad_channel = getDeadChannel(dead_channel_path)
    tetrode_geom = '/home/ubuntu/to_sort/sorting_files/geom_all_tetrodes_original.csv'
    geom = pd.read_csv(tetrode_geom,header=None).values
    recording = se.NumpyRecordingExtractor(signal, sampling_rate, geom)
    recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel)
    recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = st.preprocessing.whiten(recording)
    recording = se.CacheRecordingExtractor(recording)
    # reconstruct a sorting extractor
    times, labels = spatial_firing2label(spatial_firing)
    sorting = se.NumpySortingExtractor()
    sorting.set_times_labels(times=times, labels=labels)
    sorting.set_sampling_frequency(sampling_frequency=sampling_rate)
    st.postprocessing.export_to_phy(recording, sorting, output_folder=output_folder,
                                    copy_binary=False, ms_before=0.5, ms_after=0.5)
    print("I have created the phy output for ", recording)

def process_waveform_pca(recording, remove_outliers):
    spatial_firing_path = recording+"/MountainSort/DataFrames/spatial_firing.pkl"
    output_folder = recording+"/MountainSort/Figures/firing_properties/waveform_pca/"
    firing_properties_path = recording+"/MountainSort/Figures/firing_properties/"
    if not os.path.exists(firing_properties_path):
        os.mkdir(firing_properties_path)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if os.path.exists(spatial_firing_path):
        spatial_firing = pd.read_pickle(spatial_firing_path)
        plot_pca_waveforms(spatial_firing, recording, output_folder=output_folder, remove_outliers=remove_outliers)
    else:
        print("could not process waveform pca")
        print("spatial_firing.pkl does not exist for ", recording)


