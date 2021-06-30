import PreClustering.dead_channels
import PreClustering.make_sorting_database
import PreClustering.parameters
import mdaio
import numpy as np
import OpenEphys
import os
import control_sorting_analysis
import shutil

import file_utility
from PreClustering import convert_open_ephys_to_mda

prm = PreClustering.parameters.Parameters()


def init_params():
    prm.set_sampling_rate(30000)
    prm.set_num_tetrodes(4)
    prm.set_movement_ch('100_ADC2.continuous')
    prm.set_opto_ch('100_ADC3.continuous')
    # file_utility.init_data_file_names(prm, '105_CH', '_0')  # old files
    file_utility.init_data_file_names(prm, '100_CH', '')  # currently used (2018)
    prm.set_waveform_size(40)

    # These are not exclusive, both can be True for the same recording - that way it'll be sorted twice
    prm.set_is_tetrode_by_tetrode(False)  # set to True if you want the spike sorting to be done tetrode by tetrode
    prm.set_is_all_tetrodes_together(True)  # set to True if you want the spike sorting done on all tetrodes combined


def split_continuous_data(recording_to_sort, stitch_points):
    """
    This function is needed when multiple recordings were sorted together.
    The continuous data from all the recordings were concatenated with the first recording's. This function removes
    these from the continuous files of the first recording and saves the original files again. (The other recordings'
    raw data was not modified before sorting.)

    :param recording_to_sort: local path to recording folder
    :param stitch_points: time points where recordings were concatenated before sorting
    :return: the total length of the concatenated recordings in sampling points
    """
    dir = [f.path for f in os.scandir(recording_to_sort)]
    first_stitch_point = stitch_points[0]
    n_timestamps = 0
    for filepath in dir:
        filename = filepath.split("/")[-1]

        if filename.startswith(prm.get_continuous_file_name()):
            ch = OpenEphys.loadContinuous(recording_to_sort + '/' + filename)

            # this calculates total sample length of recordings
            if n_timestamps == 0:
                n_timestamps = len(ch["data"])

            ch['data'] = ch['data'][:first_stitch_point]
            ch['timestamps'] = ch['timestamps'][:first_stitch_point]
            ch['recordingNumber'] = ch['recordingNumber'][:first_stitch_point]
            OpenEphys.writeContinuousFile(filepath, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])
    return n_timestamps


def make_sorting_output_folder_for_paired_recording(paired_path_local, sorter_name):
    sorting_output_folder = paired_path_local + '/Electrophysiology/' + sorter_name + '/'
    if os.path.exists(sorting_output_folder) is False:
        os.makedirs(sorting_output_folder)
    return sorting_output_folder


def split_filtered_electrophysiology_data_file(recording_to_sort: str, sorter_name: str, stitch_points: list):
    """
    Split filt.mda file and move to paired recording folders. This is the filtered and whitened data that is used for
    spike detection. We use this for plotting the action potentials later on so it needs to correspond to the firing
    times of the output of the sorting.
    :param recording_to_sort: Path to recording #1 that is sorted together with other recordings
    :param sorter_name: name of spike sorting program
    :param stitch_points: time points where recordings were concatenated before sorting
    :return:
    """
    tags = control_sorting_analysis.get_tags_parameter_file(recording_to_sort)
    paired_recordings = control_sorting_analysis.check_for_paired(tags)
    # this is the concatenated filtered data file
    filtered_data_path = recording_to_sort + '/Electrophysiology/' + sorter_name + '/filt.mda'
    filtered_data = mdaio.readmda(filtered_data_path)
    for paired_index, recording in enumerate(paired_recordings):
        first_half_of_local_path = '/'.join(recording_to_sort.split('/')[:-1])
        second_half = '/' + recording.split('/')[-1]
        paired_path_local = first_half_of_local_path + second_half
        sorting_output_folder = make_sorting_output_folder_for_paired_recording(paired_path_local, sorter_name)
        # get correct part of filtered data based on stitch points
        paired_recording_filtered = filtered_data[:, stitch_points[paired_index]:stitch_points[paired_index + 1]]
        # save filtered data
        mdaio.writemda16i(paired_recording_filtered, sorting_output_folder + 'filt.mda')


def copy_curation_information(recording_to_sort: str, sorter_name: str):
    tags = control_sorting_analysis.get_tags_parameter_file(recording_to_sort)
    paired_recordings = control_sorting_analysis.check_for_paired(tags)
    # this are the cluster quality metrics
    curation_data = recording_to_sort + '/Electrophysiology/' + sorter_name + '/cluster_metrics.json'
    if os.path.exists(curation_data):
        for recording in paired_recordings:
            first_half_of_local_path = '/'.join(recording_to_sort.split('/')[:-1])
            second_half = '/' + recording.split('/')[-1]
            paired_path_local = first_half_of_local_path + second_half
            sorting_output_folder = make_sorting_output_folder_for_paired_recording(paired_path_local, sorter_name)
            shutil.copyfile(curation_data, sorting_output_folder + '/cluster_metrics.json')


def split_firing_times_sorting_output(recording_to_sort: str, sorter_name: str, stitch_points: list):
    tags = control_sorting_analysis.get_tags_parameter_file(recording_to_sort)
    paired_recordings = control_sorting_analysis.check_for_paired(tags)
    # this are the firing times of the sorted clusters
    firing_times_path = recording_to_sort + '/Electrophysiology/' + sorter_name + '/firings.mda'
    if os.path.exists(firing_times_path):
        firing_info = mdaio.readmda(firing_times_path)

        for stitch_index, recording in enumerate(paired_recordings):
            first_half_of_local_path = '/'.join(recording_to_sort.split('/')[:-1])
            second_half = '/' + recording.split('/')[-1]
            paired_path_local = first_half_of_local_path + second_half
            sorting_output_folder = make_sorting_output_folder_for_paired_recording(paired_path_local, sorter_name)
            # split firing times file based on stitch point
            after_previous_stitch = firing_info[1] > stitch_points[stitch_index]
            before_next_stitch = firing_info[1] <= stitch_points[stitch_index + 1]
            in_recording = after_previous_stitch & before_next_stitch
            indices_in_recording = np.where(in_recording == 1)[0]
            firing_times_recording = firing_info[:, indices_in_recording]
            firing_times_recording[1] -= stitch_points[stitch_index]  # shift so they start at 0
            mdaio.writemda32(firing_times_recording, sorting_output_folder + '/firings.mda')
        # split first one too
        in_recording = firing_info[1] < stitch_points[0]
        indices_in_recording = np.where(in_recording == 1)[0]
        firing_times_recording = firing_info[:, indices_in_recording]
        os.remove(firing_times_path)
        mdaio.writemda32(firing_times_recording, firing_times_path)


def split_back(recording_to_sort: str, stitch_points: list, sorter_name='MountainSort'):
    """
    :param sorter_name: name of spike sorting program
    :param recording_to_sort: Path to recording #1 that is sorted together with other recordings
    :param stitch_points: time points where recordings were concatenated before sorting
    :return: the path (same as input parameter) and the total number of time steps in the combined data
    """
    print('I will split the data that was sorted together. It might take a while.')
    n_timestamps = split_continuous_data(recording_to_sort, stitch_points)
    # split filtered data (sorting input)
    split_filtered_electrophysiology_data_file(recording_to_sort, sorter_name, stitch_points)
    # copy curation file
    copy_curation_information(recording_to_sort, sorter_name)
    # split firings.mda
    split_firing_times_sorting_output(recording_to_sort, sorter_name, stitch_points)
    return recording_to_sort, n_timestamps


def stitch_recordings(recording_to_sort: str, paired_recordings: list):
    """
    Load continuous data from multiple recordings, concatenate the arrays and write new continuous files.
    :param recording_to_sort: path to recording #1
    :param paired_recordings: path list of recordings to sort together with recording #1
    :return: combined recording and time points where a new recording started
    """
    print('I will stitch these recordings together now so they can be sorted together. It might take a while.')
    init_params()
    file_utility.set_continuous_data_path(prm)

    directory_list = [f.path for f in os.scandir(recording_to_sort)]
    stitch_points = []
    added_first_stitch = False
    added_paired_stitch = False
    for filepath in directory_list:
        filename = filepath.split("/")[-1]
        if filename.startswith(prm.get_continuous_file_name()):
            ch = OpenEphys.loadContinuous(recording_to_sort + '/' + filename)
            if not added_first_stitch:
                length_of_recording = len(ch['data'])
                stitch_points.append(length_of_recording)
                added_first_stitch = True
            for recording in paired_recordings:
                ch_p = OpenEphys.loadContinuous(recording + '/' + filename)
                ch['data'] = np.append(ch['data'], ch_p['data'])
                ch['timestamps'] = np.append(ch['timestamps'], ch_p['timestamps'])
                ch['recordingNumber'] = np.append(ch['recordingNumber'], ch_p['recordingNumber'])
                if not added_paired_stitch:
                    length_of_other_recording = len(ch_p['data'])
                    previous_stitch = stitch_points[-1]
                    stitch_points.append(previous_stitch + length_of_other_recording)
            added_paired_stitch = True
            OpenEphys.writeContinuousFile(filepath, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])

    return recording_to_sort, stitch_points


# Prepares input for running spike sorting for the recording.
def process_a_dir(dir_name):
    print('')
    print('I am pre-processing data in {} before spike sorting.'.format(dir_name))
    prm.set_date(dir_name.rsplit('/', 2)[-2])

    prm.set_filepath(dir_name)
    file_utility.set_continuous_data_path(prm)

    PreClustering.dead_channels.get_dead_channel_ids(prm)  # read dead_channels.txt
    file_utility.create_folder_structure(prm)

    if prm.get_is_tetrode_by_tetrode() is True:
        print('------------------------------------------')
        print('I am making one mda file for each tetrode.')
        print('------------------------------------------')
        PreClustering.make_sorting_database.create_sorting_folder_structure_separate_tetrodes(prm)
        convert_open_ephys_to_mda.convert_continuous_to_mda(prm)
        print('All 4 tetrodes were converted to separate mda files.')
        print('*****************************************************')

    if prm.get_is_all_tetrodes_together() is True:
        print('-------------------------------------------------------------------------')
        print('I am converting all channels into one mda file. This will take some time.')
        print('-------------------------------------------------------------------------')
        PreClustering.make_sorting_database.create_sorting_folder_structure(prm)
        convert_open_ephys_to_mda.convert_all_tetrodes_to_mda(prm)
        print('The big mda file is created, it is in Electrophysiology' + prm.get_spike_sorter())
        print('***************************************************************************************')


def pre_process_data(dir_name, sorter_name='MountainSort'):
    init_params()
    prm.set_spike_sorter(sorter_name)
    process_a_dir(dir_name + '/')
