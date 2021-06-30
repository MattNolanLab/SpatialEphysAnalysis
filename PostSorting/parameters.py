import os

class Parameters:

    is_ubuntu = True
    is_windows = False
    is_stable = False
    delete_two_min = False
    first_half_only = False
    second_half_only = False
    pixel_ratio = None
    opto_channel = ''
    sync_channel = ''
    sampling_rate = 0
    opto_tagging_start_index = None
    sampling_rate_rate = 0
    local_recording_folder_path = ''
    file_path = []
    output_path = []
    ms_tmp_path = []
    total_length_sampling_points = 0
    dead_channels = []
    sorter_name = []
    stitchpoint = None
    paired_order = None
    ephys_channels = []
    shared_ephys_channel_marker = "100_CH"

    # vr parameters
    first_trial_channel = ''  # vr
    second_trial_channel = ''  # vr
    movement_channel = ''  # vr
    stop_threshold = 10.7  # vr
    track_length = 200  # vr
    cue_conditioned_goal = False
    cue_goal_min = -10
    cue_goal_max = 10
    vr_grid_analysis_bin_size = 20

    def __init__(self):
        return

    def get_sorter_name(self):
            return Parameters.sorter_name

    def set_sorter_name(self, name):
        Parameters.sorter_name = name

    def get_first_half_only(self):
        return Parameters.first_half_only

    def set_first_half_only(self, is_first):
        Parameters.first_half_only = is_first

    def get_second_half_only(self):
        return Parameters.second_half_only

    def set_second_half_only(self, is_second):
        Parameters.second_half_only = is_second

    def get_pixel_ratio(self):
        return Parameters.pixel_ratio

    def set_pixel_ratio(self, pr):
        Parameters.pixel_ratio = pr

    def get_opto_channel(self):
        return Parameters.opto_channel

    def set_opto_channel(self, opto_ch):
        Parameters.opto_channel = opto_ch

    def get_sync_channel(self):
        return Parameters.sync_channel

    def set_sync_channel(self, sync_ch):
        Parameters.sync_channel = sync_ch

    def get_ephys_channels(self):
        return Parameters.ephys_channels

    def set_ephys_channels(self, ephys_channels):
        Parameters.ephys_channels = ephys_channels

    def get_shared_ephys_channel_marker(self):
        return Parameters.shared_ephys_channel_marker

    def set_shared_ephys_channel_marker(self, shared_ephys_channel_marker):
        Parameters.shared_ephys_channel_marker = shared_ephys_channel_marker


    def get_sampling_rate(self):
        return Parameters.sampling_rate

    def set_sampling_rate(self, sr):
        Parameters.sampling_rate = sr

    def get_downsampled_rate(self):
        return Parameters.downsampled_rate

    def set_downsampled_rate(self, dsr):
        Parameters.downsampled_rate = dsr

    def get_local_recording_folder_path(self):
        return Parameters.local_recording_folder_path

    def set_local_recording_folder_path(self, path):
        Parameters.local_recording_folder_path = path

    def get_filepath(self):
        return Parameters.file_path

    def set_file_path(self, path):
        Parameters.file_path = path

    def get_output_path(self):
        return Parameters.output_path

    def set_output_path(self, path):
        Parameters.output_path = path

    def get_ms_tmp_path(self):
        return Parameters.ms_tmp_path

    def set_ms_tmp_path(self, path):
        Parameters.ms_tmp_path = path

    def get_dead_channels(self):
        return Parameters.dead_channels

    def set_dead_channels(d_ch = [], *args):
        dead_ch = []
        Parameters.dead_channels = dead_ch

        for dead_chan in args:
            dead_ch.append(dead_chan)

        Parameters.dead_channels = dead_ch

    def set_dead_channel_from_txt_file(self, dead_channel_txt_file_path):

        if os.path.isfile(dead_channel_txt_file_path) is True:
            if os.stat(dead_channel_txt_file_path).st_size == 0:
                print("theres a dead channel file but no dead channel is given")
            else:
                dead_channel_reader = open(dead_channel_txt_file_path, 'r')
                dead_channels = dead_channel_reader.readlines()
                dead_channels = list([x.strip() for x in dead_channels])
                Parameters.dead_channels = dead_channels


    def get_dead_channel_path(self):
        return Parameters.dead_channel_path

    def set_dead_channel_path(self, dead_ch):
        Parameters.dead_channel_path = dead_ch

    #######################################################
    # Parameters specific to VR
    def get_first_trial_channel(self):
        return Parameters.first_trial_channel


    def set_first_trial_channel(self, first_trial_channel):
        Parameters.first_trial_channel = first_trial_channel


    def get_second_trial_channel(self):
        return Parameters.second_trial_channel


    def set_second_trial_channel(self, second_trial_channel):
        Parameters.second_trial_channel = second_trial_channel


    def get_movement_channel(self):
        return Parameters.movement_channel


    def set_movement_channel(self, movement_channel):
        Parameters.movement_channel = movement_channel


    def get_track_length(self):
        return Parameters.track_length

    def set_track_length(self, track_length):
        Parameters.track_length = float(track_length)

    def get_stop_threshold(self):
        return Parameters.stop_threshold


    def set_stop_threshold(self, st):
        Parameters.stop_threshold = st

    def get_delete_two_minutes(self):
        return Parameters.delete_two_min

    def set_delete_two_minutes(self, delete_two_minutes):
        Parameters.delete_two_min = delete_two_minutes

    def get_cue_conditioned_goal(self):
        return Parameters.cue_conditioned_goal

    def set_cue_conditioned_goal(self, cue_conditioned_goal):
        Parameters.cue_conditioned_goal = cue_conditioned_goal

    def get_goal_location_channel(self):
        return Parameters.goal_location_channel

    def set_goal_location_chennl(self, goal_location_channel):
        Parameters.goal_location_channel = goal_location_channel

    def set_vr_grid_analysis_bin_size(self, bin_size):
        Parameters.vr_grid_analysis_bin_size = bin_size

    def get_vr_grid_analysis_bin_size(self):
        return Parameters.vr_grid_analysis_bin_size
