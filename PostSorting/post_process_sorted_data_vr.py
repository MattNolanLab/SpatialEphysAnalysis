import os
import PostSorting.curation
import PostSorting.load_firing_data
import PostSorting.load_snippet_data
import PostSorting.parameters
import PostSorting.temporal_firing
import PostSorting.vr_spatial_data
import PostSorting.vr_make_plots
import PostSorting.vr_spatial_firing
import PostSorting.make_plots
import PostSorting.vr_sync_spatial_data
import PostSorting.vr_firing_rate_maps
import PostSorting.vr_FiringMaps_InTime
import PostSorting.vr_speed_analysis
import gc
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_grid_cells
import PostSorting.lfp
import settings

prm = PostSorting.parameters.Parameters()


def initialize_parameters(recording_to_process):
    prm.set_sampling_rate(30000)
    prm.set_downsampled_rate(1000)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_opto_channel('100_ADC3.continuous')
    prm.set_stop_threshold(0.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_movement_channel('100_ADC2.continuous')
    prm.set_first_trial_channel('100_ADC4.continuous')
    prm.set_second_trial_channel('100_ADC5.continuous')
    prm.set_goal_location_chennl('100_ADC7.continuous')
    prm.set_ephys_channels(PostSorting.load_firing_data.available_ephys_channels(recording_to_process, prm))
    prm.set_file_path(recording_to_process)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_ms_tmp_path('/tmp/mountainlab/')


def process_position_data(recording_to_process, output_path, track_length, stop_threshold):
    raw_position_data, position_data = PostSorting.vr_sync_spatial_data.syncronise_position_data(recording_to_process, output_path, track_length)
    processed_position_data = PostSorting.vr_spatial_data.process_position(raw_position_data, stop_threshold,track_length)
    return raw_position_data, processed_position_data, position_data


def process_firing_properties(recording_to_process, sorter_name, dead_channels, total_length_seconds, opto_tagging_start_index=None):
    spike_data = PostSorting.load_firing_data.process_firing_times(recording_to_process, sorter_name, dead_channels, opto_tagging_start_index)
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, total_length_seconds)
    return spike_data


def save_data_frames(output_path, spatial_firing_movement=None, spatial_firing_stationary=None, spatial_firing=None,
                     raw_position_data=None, processed_position_data=None, position_data=None, snippet_data=None, bad_clusters=None,
                     lfp_data=None):
    if os.path.exists(output_path + '/DataFrames') is False:
        os.makedirs(output_path + '/DataFrames')
    if spatial_firing_movement is not None:
        spatial_firing_movement.to_pickle(output_path + '/DataFrames/spatial_firing_movement.pkl')
    if spatial_firing_stationary is not None:
        spatial_firing_stationary.to_pickle(output_path + '/DataFrames/spatial_firing_stationary.pkl')
    if spatial_firing is not None:
        spatial_firing.to_pickle(output_path + '/DataFrames/spatial_firing.pkl')
    if raw_position_data is not None:
        print(" I am not saving the raw positional pickle at the moment")
        #raw_position_data.to_pickle(output_path + '/DataFrames/raw_position_data.pkl')
    if processed_position_data is not None:
        processed_position_data.to_pickle(output_path + '/DataFrames/processed_position_data.pkl')
    if position_data is not None:
        position_data.to_pickle(output_path+ '/DataFrames/position_data.pkl')
    if bad_clusters is not None:
        bad_clusters.to_pickle(output_path+ '/DataFrames/noisy_clusters.pkl')
    if snippet_data is not None:
        snippet_data.to_pickle(output_path + '/DataFrames/snippet_data.pkl')
    if lfp_data is not None:
        lfp_data.to_pickle(output_path + "/DataFrames/lfp_data.pkl")

def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')
    if os.path.exists(recording_to_process + '/DataFrames') is False:
        os.makedirs(recording_to_process + '/DataFrames')
    if os.path.exists(recording_to_process + '/Firing_fields') is False:
        os.makedirs(recording_to_process + '/Firing_fields')
    if os.path.exists(recording_to_process + '/Data_test') is False:
        os.makedirs(recording_to_process + '/Data_test')

def process_running_parameter_tag(running_parameter_tags):
    stop_threshold = 4.9  # defaults
    track_length = 200 # default assumptions
    cue_conditioned_goal = False

    if not running_parameter_tags:
        return stop_threshold, track_length, cue_conditioned_goal

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag.startswith('stop_threshold'):
            stop_threshold = float(tag.split("=")[1])
        elif tag.startswith('track_length'):
            track_length = int(tag.split("=")[1])
        elif tag.startswith('cue_conditioned_goal'):
            cue_conditioned_goal = bool(tag.split('=')[1])
        else:
            print('Unexpected / incorrect tag in the third line of parameters file')
            unexpected_tag = True
    return stop_threshold, track_length, cue_conditioned_goal

def post_process_recording(recording_to_process, session_type, running_parameter_tags=False,
                           sorter_name=settings.sorterName):

    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    output_path = recording_to_process+'/'+settings.sorterName


    # Keep all parameter object reference at this level, do not pass them beyond this level
    # keep the configuration of the prm object at a single location only for easily tracking
    stop_threshold, track_length, cue_conditioned_goal = process_running_parameter_tag(running_parameter_tags)
    prm.set_stop_threshold(stop_threshold)
    prm.set_track_length(track_length)
    prm.set_vr_grid_analysis_bin_size(5)
    prm.set_cue_conditioned_goal(cue_conditioned_goal)

    dead_channels = prm.get_dead_channels()
    ephys_channels = prm.get_ephys_channels()

    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())

    # Process LPF
    lfp_data = PostSorting.lfp.process_lfp(recording_to_process, ephys_channels, output_path, dead_channels)
    # Process position
    raw_position_data, processed_position_data, position_data = process_position_data(recording_to_process, output_path, track_length, stop_threshold)
    total_length_seconds = raw_position_data.time_seconds.values[-1]

    # Process firing
    spike_data = process_firing_properties(recording_to_process, sorter_name, dead_channels, total_length_seconds)

    # Curation
    spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, sorter_name, prm.get_local_recording_folder_path(), prm.get_ms_tmp_path())

    # Get snippet of spike waveforms
    snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording_to_process, sorter_name, dead_channels, random_snippets=False)

    # Perform experiment related analysis
    if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
        PostSorting.vr_make_plots.make_plots(processed_position_data, spike_data=None,
                                             output_path=output_path, track_length=track_length)

        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        print('No curated clusters found. Saving dataframe for noisy clusters...')
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
    else:
      
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        print(str(len(spike_data)), ' curated clusters found. Processing spatial firing...')
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')

        spike_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording_to_process, sorter_name, dead_channels, random_snippets=True)
        spike_data_movement, spike_data_stationary, spike_data = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data)
        #spike_data = PostSorting.vr_grid_cells.process_vr_grid(spike_data, position_data, prm.get_vr_grid_analysis_bin_size(), prm)
        spike_data = PostSorting.vr_firing_rate_maps.make_firing_field_maps(spike_data, processed_position_data, settings.vr_bin_size_cm, track_length)
        #spike_data = PostSorting.vr_FiringMaps_InTime.control_convolution_in_time(spike_data, raw_position_data)
        spike_data = PostSorting.theta_modulation.calculate_theta_index(spike_data, output_path, settings.sampling_rate)

        PostSorting.vr_make_plots.make_plots(processed_position_data, spike_data=spike_data,
                                             output_path=output_path, track_length=track_length)

    save_data_frames(output_path,
                     spatial_firing=spike_data,
                     raw_position_data=raw_position_data,
                     processed_position_data=processed_position_data,
                     position_data=position_data,
                     snippet_data=snippet_data,
                     bad_clusters=bad_clusters,
                     lfp_data=lfp_data)
    gc.collect()


