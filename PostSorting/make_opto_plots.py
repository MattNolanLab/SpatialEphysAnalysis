import glob
import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import PostSorting.parameters
import PostSorting.make_plots
import matplotlib.image as mpimg
import open_ephys_IO
import mdaio


def make_folder_for_figures(output_path):
    save_path = output_path + '/Figures/opto_stimulation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path


def get_first_spikes(cluster_rows, stimulation_start, sampling_rate, latency_window_ms=10):
    '''
    Find first spikes after the light pulse in a 10ms window
    :param cluster_rows: Opto stimulation trials corresponding to cluster (binary array)
    :param stimulation_start: Index (in samplin rate) where the stimulation starts in the peristimulus array)
    :param sampling_rate: Sampling rate of electrophysiology data
    :param latency_window_ms: The time window used to calculate latencies in ms (spikes outside this are not included)
    :return:
    '''

    latency_window = latency_window_ms * sampling_rate / 1000  # this is the latency window in sampling points
    events_after_stimulation = cluster_rows[
        cluster_rows.columns[int(stimulation_start):int(stimulation_start + latency_window)]]
    # events_after_stimulation = cluster_rows[cluster_rows.columns[int(stimulation_start):]]
    spikes = np.array(events_after_stimulation).astype(int) == 1
    first_spikes = spikes.cumsum(axis=1).cumsum(axis=1) == 1
    zeros_left = np.zeros((spikes.shape[0], int(stimulation_start - 1)))
    first_spikes = np.hstack((zeros_left, first_spikes))
    zeros_right = np.zeros((spikes.shape[0], int(cluster_rows.shape[1] - stimulation_start - sampling_rate / 100)))
    first_spikes = np.hstack((first_spikes, zeros_right))
    sample_times_firsts = np.argwhere(first_spikes)[:, 1]
    trial_numbers_firsts = np.argwhere(first_spikes)[:, 0]
    return sample_times_firsts, trial_numbers_firsts


def plot_spikes_around_light(ax, cluster_rows, sampling_rate, light_pulse_duration, latency_window_ms):
    sample_times = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 1]
    trial_numbers = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 0]
    stimulation_start = cluster_rows.shape[1] / 2  # the peristimulus array is made so the pulse starts in the middle
    stimulation_end = cluster_rows.shape[1] / 2 + light_pulse_duration

    ax.axvspan(stimulation_start, stimulation_end, 0, cluster_rows.shape[0], alpha=0.5, color='lightblue')
    ax.vlines(x=sample_times, ymin=trial_numbers, ymax=(trial_numbers + 1), color='black', zorder=2, linewidth=3)
    sample_times_firsts, trial_numbers_firsts = get_first_spikes(cluster_rows, stimulation_start, sampling_rate,
                                                                 latency_window_ms)
    ax.vlines(x=sample_times_firsts, ymin=trial_numbers_firsts, ymax=(trial_numbers_firsts + 1), color='red', zorder=2,
              linewidth=3)


def format_peristimulus_plot(positions, sampling_rate):
    """
    Add axis labels and set size of figures.
    """
    plt.cla()
    peristimulus_figure, ax = plt.subplots()
    peristimulus_figure.set_size_inches(5, 5, forward=True)
    plt.xlabel('Time (ms)', fontsize=24)
    labels = np.array(positions) / sampling_rate * 1000  # convert sampling points to ms
    labels = (str(int(labels[0])), str(int(labels[1])), str(int(labels[2])))
    plt.xticks(positions, labels)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    return peristimulus_figure, ax


def get_binary_peristimulus_data_for_cluster(peristimulus_spikes: pd.DataFrame, cluster: str):
    """
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    :param cluster: cluster id
    :return: rows of data frame that correspond to cluster and only the columns that contain the binary spike data
    """

    cluster_rows_boolean = peristimulus_spikes.cluster_id.astype(int) == int(cluster)
    cluster_rows_annotated = peristimulus_spikes[cluster_rows_boolean]
    cluster_rows = cluster_rows_annotated.iloc[:, 2:]
    return cluster_rows


def plot_peristimulus_raster_for_cluster(peristimulus_spikes, cluster, session, sampling_rate, light_pulse_duration,
                                         latency_window_ms, save_path):
    cluster_rows = get_binary_peristimulus_data_for_cluster(peristimulus_spikes, cluster)
    positions = [0, cluster_rows.shape[1]/2, cluster_rows.shape[1]]
    peristimulus_figure, ax = format_peristimulus_plot(positions, sampling_rate)
    plot_spikes_around_light(ax, cluster_rows, sampling_rate, light_pulse_duration, latency_window_ms)
    plt.ylim(0, cluster_rows.shape[0])
    plt.xlim(0, cluster_rows.shape[1])
    plt.ylabel('Trial', fontsize=24)
    plt.yticks(np.arange(0, cluster_rows.shape[0] + 1, 50))  # show every 50th tick only
    plt.tight_layout()
    plt.savefig(save_path + '/peristimulus_raster_' + session.iloc[0] + '_' + str(cluster) + '.png', dpi=300)
    plt.close()


# do not use this on data from more than one session
def plot_peristimulus_raster(peristimulus_spikes: pd.DataFrame, output_path: str, sampling_rate: int,
                             light_pulse_duration: int, latency_window_ms: int):
    """
    PLots spike raster from light stimulation trials around the light. The plot assumes that the stimulation
    starts in the middle of the peristimulus_spikes array.
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    :param output_path: fist half of the path where the plot is saved
    :param sampling_rate: sampling rate of electrophysiology data
    :param light_pulse_duration: duration of light pulse (ms)
    :param latency_window_ms: time window where spikes are considered evoked for a given trial
    """

    # make sure it's a single session
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = make_folder_for_figures(output_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        plot_peristimulus_raster_for_cluster(peristimulus_spikes, cluster, peristimulus_spikes.session_id, sampling_rate, light_pulse_duration,
                                             latency_window_ms, save_path)


def get_latencies_for_cluster(spatial_firing, cluster_id):
    cluster = spatial_firing[spatial_firing.cluster_id == int(cluster_id)]
    latencies_mean = np.round(cluster.opto_latencies_mean_ms, 2)
    latencies_sd = np.round(cluster.opto_latencies_sd_ms, 2)
    if len(latencies_mean) > 0:
        return pd.to_numeric(latencies_mean).iloc[0], pd.to_numeric(latencies_sd).iloc[0]
    else:
        return pd.to_numeric(latencies_mean), pd.to_numeric(latencies_sd)


def convert_y_axis_to_hz(cluster_rows, sampling_rate, number_of_histogram_bins, hist):
    window_size_seconds = cluster_rows.shape[1] / sampling_rate
    bin_size_seconds = window_size_seconds / number_of_histogram_bins
    hist = hist / bin_size_seconds
    hist = hist / cluster_rows.shape[0]  # also divide by number of trials
    return hist


def make_peristimulus_histogram_for_cluster(spatial_firing, peristimulus_spikes, cluster, session, light_pulse_duration,
                                            save_path, sampling_rate, middle_only, y_axis_in_hz=False):
    number_of_histogram_bins = 100
    cluster_rows = get_binary_peristimulus_data_for_cluster(peristimulus_spikes, cluster)
    cluster_rows = cluster_rows.astype(int).to_numpy()
    if middle_only:
        middle = int(cluster_rows.shape[1] / 2)
        twenty_ms = int(sampling_rate * 20 / 1000)
        cluster_rows = cluster_rows[:, middle-twenty_ms:middle + twenty_ms]
    positions = [0, cluster_rows.shape[1]/2, cluster_rows.shape[1]]
    peristimulus_figure, ax = format_peristimulus_plot(positions, sampling_rate)
    number_of_spikes_per_sampling_point = np.array(np.sum(cluster_rows, axis=0))
    stimulation_start = cluster_rows.shape[1] / 2  # stimulus pulse starts in the middle of the array
    stimulation_end = cluster_rows.shape[1] / 2 + light_pulse_duration
    latencies_mean, latencies_sd = get_latencies_for_cluster(spatial_firing, cluster)
    salt_p = np.round(spatial_firing[spatial_firing.cluster_id == int(cluster)].SALT_p.iloc[0][0], 4)
    salt_i = np.round(spatial_firing[spatial_firing.cluster_id == int(cluster)].SALT_I.iloc[0][0], 4)
    ax.axvspan(stimulation_start, stimulation_end, 0, np.max(number_of_spikes_per_sampling_point), alpha=0.5,
               color='lightblue')
    # convert to indices so we can make histogram
    spike_indices = np.where(cluster_rows.flatten() == 1)[0] % len(number_of_spikes_per_sampling_point)
    hist, bins = np.histogram(spike_indices, bins=number_of_histogram_bins)
    y_label = 'Spike count'
    if y_axis_in_hz:
        hist, y_label = convert_y_axis_to_hz(cluster_rows, sampling_rate, number_of_histogram_bins, hist)
        y_label = 'Firing rate (Hz)'
    width = 0.9 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='grey', alpha=0.5)
    plt.xlim(0, len(number_of_spikes_per_sampling_point))
    plt.ylabel(y_label, fontsize=24)
    plt.title('Mean latency: ' + str(latencies_mean) + ' ms, sd = ' + str(latencies_sd) + "\n" + ' SALT p = ' + str(salt_p) + ' SALT I = ' + str(salt_i))
    plt.tight_layout()
    if not middle_only:
        plt.savefig(save_path + '/peristimulus_histogram_' + session.iloc[0] + '_' + str(cluster) + '.png', dpi=300)
    else:
        plt.savefig(save_path + '/peristimulus_histogram_zoom' + session.iloc[0] + '_' + str(cluster) + '.png', dpi=300)
    plt.close()


def plot_peristimulus_histogram(spatial_firing: pd.DataFrame, peristimulus_spikes: pd.DataFrame, output_path: str,
                                sampling_rate: int, light_pulse_duration: int, y_axis_in_hz=False):
    """
    PLots histogram of spikes from light stimulation trials around the light. The plot assumes that the stimulation
    starts in the middle of the peristimulus_spikes array.
    :param y_axis_in_hz: instead of number of spikes, show data in spike /sec (Hz) on the y axis
    :param sampling_rate: sampling rate of electrophysiology data
    :param spatial_firing: Data frame with firing data for each cluster
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    :param output_path: fist half of the path where the plot is saved
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    :param light_pulse_duration: duration of light pulse (ms)
    """
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = make_folder_for_figures(output_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        make_peristimulus_histogram_for_cluster(spatial_firing, peristimulus_spikes, cluster, peristimulus_spikes.session_id, light_pulse_duration,
                                                save_path, sampling_rate, middle_only=False, y_axis_in_hz=y_axis_in_hz)
        make_peristimulus_histogram_for_cluster(spatial_firing, peristimulus_spikes, cluster, peristimulus_spikes.session_id, light_pulse_duration,
                                                save_path, sampling_rate, middle_only=True, y_axis_in_hz=y_axis_in_hz)


def plot_waveforms_opto(spike_data, output_path, snippets_column_name='random_snippets_opto', title='Random snippets'):
    if snippets_column_name in spike_data:
        print('I will plot the waveform shapes for each cluster for opto_tagging data.')
        save_path = output_path + '/Figures/opto_stimulation'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
            cluster_df = spike_data[(spike_data.cluster_id == cluster_id)]  # dataframe for that cluster

            max_channel = cluster_df['primary_channel'].iloc[0]
            fig = plt.figure(figsize=(5, 5))
            plt.suptitle(title, fontsize=24)
            grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)
            for channel in range(4):
                PostSorting.make_plots.plot_spikes_for_channel_centered(grid, spike_data, cluster_id, channel,
                                                                        snippets_column_name)

            plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(
                cluster_id) + '_' + snippets_column_name + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()


def add_opto_subplot(figure_path, grid, grid_row, grid_columns):
    if os.path.exists(figure_path):
        figure_name = mpimg.imread(figure_path)
        plot_name = plt.subplot(grid[grid_row, grid_columns])
        plot_name.axis('off')
        plot_name.imshow(figure_name)


def add_subplots_to_combined_opto_figure(grid, waveforms_cell_all, waveforms_opto_random, waveforms_first_spikes, peristimulus_raster,peristimulus_histogram):
    add_opto_subplot(waveforms_cell_all, grid, 0, 0)
    add_opto_subplot(waveforms_opto_random, grid, 0, 1)
    add_opto_subplot(waveforms_first_spikes, grid, 0, 2)
    add_opto_subplot(peristimulus_raster, grid, 1, 0)
    add_opto_subplot(peristimulus_histogram, grid, 1, 1)


def make_combined_opto_plot(spatial_firing, output_path):
    print('I will make the combined images for opto stimulation analysis results now.')
    save_path = output_path + '/Figures/opto_plots_combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = output_path + '/Figures/'
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)]  # data frame for that cluster
        cluster_path_name = cluster_df['session_id'].iloc[0] + '_' + str(cluster_id)
        waveforms_cell_all = figures_path + 'firing_properties/' + cluster_path_name + '_waveforms.png'
        waveforms_opto_random = figures_path + 'opto_stimulation/' + cluster_path_name + '_random_snippets_opto.png'
        waveforms_first_spikes = figures_path + 'opto_stimulation/' + cluster_path_name + '_random_first_spike_snippets_opto.png'
        peristimulus_raster = figures_path + 'opto_stimulation/peristimulus_raster_' + cluster_path_name + '.png'
        peristimulus_histogram = figures_path + 'opto_stimulation/peristimulus_histogram_' + cluster_path_name + '.png'

        number_of_rows = 2
        number_of_columns = 3
        grid = plt.GridSpec(number_of_rows, number_of_columns, wspace=0.2, hspace=0.2)
        plt.suptitle('Light responses')

        add_subplots_to_combined_opto_figure(grid, waveforms_cell_all, waveforms_opto_random, waveforms_first_spikes,
                                             peristimulus_raster, peristimulus_histogram)

        plt.savefig(save_path + '/combined_opto_' + cluster_path_name + '.png', dpi=1000)
        plt.close()


def sort_folder_names(list_of_names):
    list_of_names.sort(key=lambda x: int(x.split('CH')[1].split('.')[0]))
    return list_of_names


def load_all_channels(output_path):
    all_channels = False
    is_loaded = False
    path = '/'.join(i for i in output_path.split('/')[:-1]) + '/'
    is_first = True
    channel_count = 0
    sorted_list_of_folders = sort_folder_names(glob.glob(path + '/*CH*continuous'))
    for file_path in sorted_list_of_folders:
        if os.path.exists(file_path):
            channel_data = open_ephys_IO.get_data_continuous(file_path).astype(np.int16)
            if is_first:
                all_channels = np.zeros((len(list(glob.glob(path + '/*CH*continuous'))), channel_data.size), np.int16)
                is_first = False
                is_loaded = True
            all_channels[channel_count, :] = channel_data
            channel_count += 1
    return all_channels, is_loaded


def load_filtered_data(output_path, sorter_name='MountainSort'):
    is_loaded = False
    filtered_data = []
    file_path = '/'.join(i for i in output_path.split('/')[:-1]) + '/'
    filtered_data_path = file_path + '/Electrophysiology/' + sorter_name + '/filt.mda'
    if os.path.exists(filtered_data_path):
        filtered_data = mdaio.readmda(filtered_data_path)
        is_loaded = True
    return filtered_data, is_loaded


def get_y_axis_positions_and_labels(baselines):
    positions = []
    labels = []
    for baseline in baselines:
        positions.append(baseline - 200)
        labels.append('-200')
        positions.append(baseline)
        labels.append('0')
        positions.append(baseline + 200)
        labels.append('200')
    return positions, labels


def plot_lfp_traces(peristimulus_lfp):
    baselines = []
    baseline = 0
    for channel in range(peristimulus_lfp.shape[0]):
        plt.axhline(baseline, linestyle='--', alpha=0.6, color='grey')
        plt.plot(peristimulus_lfp[channel] + baseline, color='black')
        baselines.append(baseline)
        baseline += 600
    return baselines


def plot_light_pulse(ax, peristimulus_lfp, baselines, half_window, pulse_length):
    lowest_value = peristimulus_lfp[0].min()
    highest_value = peristimulus_lfp[-1].max() + baselines[-1]

    ax.axvspan(half_window, half_window + pulse_length, lowest_value, highest_value, alpha=0.95,
               color='lightblue')


def format_axis_labels(positions, sampling_rate):
    labels = np.array(positions) / sampling_rate * 1000  # convert sampling points to ms
    labels = (str(int(labels[0])), str(int(labels[1])), str(int(labels[2])))
    plt.xticks(positions, labels)
    plt.xlabel('Time (ms)')
    # positions, labels = get_y_axis_positions_and_labels(baselines)
    # plt.yticks(positions, labels)
    plt.ylabel('Voltage (mV)')


def make_lfp_plots_for_pulses(opto_pulses, all_channels, half_window, pulse_length, window_size, sampling_rate, path, number_of_samples=20):
    if len(opto_pulses.opto_start_times) < number_of_samples:
        number_of_samples = len(opto_pulses.opto_start_times)
    random_pulses = opto_pulses.opto_start_times.sample(n=number_of_samples, random_state=666)  # set seed
    for pulse_index, pulse in random_pulses.iteritems():
        peristimulus_lfp = all_channels[:, pulse - half_window:pulse + half_window]
        peristimulus_figure, ax = plt.subplots()
        baselines = plot_lfp_traces(peristimulus_lfp)
        plot_light_pulse(ax, peristimulus_lfp, baselines, half_window, pulse_length)
        positions = [0, half_window, window_size]
        format_axis_labels(positions, sampling_rate)
        plt.savefig(path + 'peristim_lfp_' + str(pulse_index) + '.png')
        plt.close()


def plot_lfp_around_stimulus(output_path, window_size=6000, length_of_pulse=30, sampling_rate=30000):
    output_figure_path = output_path + '/Figures/peristimulus_lfp/'
    if not os.path.exists(output_figure_path):
        os.mkdir(output_figure_path)
    half_of_window = int(window_size / 2)
    all_channels, is_loaded = load_all_channels(output_path)
    if not is_loaded:
        return
    opto_pulses_path = output_path + '/DataFrames/opto_pulses.pkl'
    if os.path.exists(opto_pulses_path):
        opto_pulses = pd.read_pickle(opto_pulses_path)
        # the times in opto_pulses are relative to the combined data
        make_lfp_plots_for_pulses(opto_pulses, all_channels, half_of_window, length_of_pulse, window_size, sampling_rate, output_figure_path)


def plot_filtered_lfp_around_stimulus(output_path, window_size=6000, length_of_pulse=30, sampling_rate=30000):
    output_figure_path = output_path + '/Figures/peristimulus_lfp_filtered/'
    if not os.path.exists(output_figure_path):
        os.mkdir(output_figure_path)
    half_of_window = int(window_size / 2)
    # this is the data that is used for spike detection
    all_channels, is_loaded = load_filtered_data(output_path)
    if not is_loaded:
        return
    opto_pulses_path = output_path + '/DataFrames/opto_pulses.pkl'
    if os.path.exists(opto_pulses_path):
        opto_pulses = pd.read_pickle(opto_pulses_path)
        make_lfp_plots_for_pulses(opto_pulses, all_channels, half_of_window, length_of_pulse, window_size, sampling_rate, output_figure_path)


def make_optogenetics_plots(spatial_firing: pd.DataFrame, output_path: str, sampling_rate: int):
    """
    :param paired_order: number of recording in series if multiple are sorted together
    :param stitch_point: list of points where recordings are stitched together
    :param spatial_firing: data frame where each row corresponds to a cluster
    :param output_path: output folder to save figures in (usually /MountainSort)
    :param sampling_rate: sampling rate of electrophysiology data
    """

    peristimulus_spikes_path = output_path + '/DataFrames/peristimulus_spikes.pkl'
    opto_parameters_path = output_path + '/DataFrames/opto_parameters.pkl'
    if os.path.exists(peristimulus_spikes_path):
        if os.path.exists(opto_parameters_path):
            opto_parameters = pd.read_pickle(opto_parameters_path)
            light_pulse_duration = opto_parameters.duration.iloc[0] * sampling_rate / 1000
            latency_window_ms = opto_parameters.first_spike_latency_ms.iloc[0]
        else:
            print('There is no metadata saved for optical stimulation. I will assume the pulses are 3 ms and that '
                  'the latencies should be calculated in a 10ms window.')
            light_pulse_duration = 90
            latency_window_ms = 10

        # binary array containing light stimulation trials in each row (0 means no spike 1 means spike at a sampling point)
        peristimulus_spikes = pd.read_pickle(peristimulus_spikes_path)
        plot_lfp_around_stimulus(output_path)
        plot_filtered_lfp_around_stimulus(output_path, window_size=6000, length_of_pulse=30, sampling_rate=30000)
        plot_peristimulus_raster(peristimulus_spikes, output_path, sampling_rate, light_pulse_duration=light_pulse_duration,
                                 latency_window_ms=latency_window_ms)
        plot_peristimulus_histogram(spatial_firing, peristimulus_spikes, output_path, sampling_rate, light_pulse_duration=light_pulse_duration)
        plot_waveforms_opto(spatial_firing, output_path, snippets_column_name='random_snippets_opto', title='During opto-tagging')
        plot_waveforms_opto(spatial_firing, output_path, snippets_column_name='random_first_spike_snippets_opto', title='First spikes after light')
        make_combined_opto_plot(spatial_firing, output_path)


