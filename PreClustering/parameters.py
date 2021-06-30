'''
This class has the setters and getters for the parameters. The parameters need to be set in main
and they can be called by calling the getter functions.
For example to set and get filepath:
prm.set_filepath('C:/Users/s1466507/Documents/Ephys/deep MEC/day3/2016-08-29_09-15-54/')
prm.get_filepath()
'''


class Parameters:
    analyze_tetrode_by_tetrode = False
    analyze_all_tetrodes_together = True

    sorter_name = 'MountainSort'

    dead_channels = []

    filepath = ''
    dead_channel_path = ''
    continuous_file_name = ''
    continuous_file_name_end = ''
    date = ''
    behaviour_analysis_path = ''
    behaviour_data_path = ''
    behaviour_path = ''

    spike_path = ''
    ephys_path = ''

    sampling_rate = 30000
    num_tetrodes = 0
    movement_ch = '100_ADC2.continuous'
    opto_ch = '100_ADC3.continuous'
    waveform_size = 40


    def __init__(self):
        return

    def get_spike_sorter(self):
        return Parameters.sorter_name

    def set_spike_sorter(self, sorter_nme):
        Parameters.sorter_name = sorter_nme

    def get_is_tetrode_by_tetrode(self):
        return Parameters.analyze_tetrode_by_tetrode

    def set_is_tetrode_by_tetrode(self, is_tetrode_by_tetrode):
        Parameters.analyze_tetrode_by_tetrode = is_tetrode_by_tetrode

    def get_is_all_tetrodes_together(self):
        return Parameters.analyze_all_tetrodes_together

    def set_is_all_tetrodes_together(self, all_tetrodes_together):
        Parameters.analyze_all_tetrodes_together = all_tetrodes_together

    def get_dead_channels(self):
        return Parameters.dead_channels

    def set_dead_channels(d_ch = [], *args):
        dead_ch = []
        for dead_chan in args:
            dead_ch.append(dead_chan)

        Parameters.dead_channels = dead_ch


    def get_date(self):
        return Parameters.date

    def set_date(self, dt):
        Parameters.date = dt

    def get_filepath(self):
        return Parameters.filepath

    def set_filepath(self, fp):
        Parameters.filepath = fp

    def get_dead_channel_path(self):
        return Parameters.dead_channel_path

    def set_dead_channel_path(self, dead_ch):
        Parameters.dead_channel_path = dead_ch


    def get_continuous_file_name(self):
        return self.continuous_file_name

    def set_continuous_file_name(self, cont_name):
        Parameters.continuous_file_name = cont_name

    def get_continuous_file_name_end(self):
            return self.continuous_file_name_end

    def set_continuous_file_name_end(self, cont_name):
        Parameters.continuous_file_name_end = cont_name

    def get_behaviour_path(self):
            return self.behaviour_path

    def set_behaviour_path(self, fn):
        Parameters.behaviour_path = fn

    def get_behaviour_data_path(self):
        return self.behaviour_data_path

    def set_behaviour_data_path(self, fn):
        Parameters.behaviour_data_path = fn

    def get_behaviour_analysis_path(self):
        return self.behaviour_analysis_path

    def set_behaviour_analysis_path(self, fn):
        Parameters.behaviour_analysis_path = fn


    def get_ephys_path(self):
        return self.ephys_path

    def set_ephys_path(self, fn):
        Parameters.ephys_path = fn

    def get_spike_path(self):
        return self.spike_path

    def set_spike_path(self, sp):
        Parameters.spike_path = sp

    def get_num_tetrodes(self):
        return self.num_tetrodes

    def set_num_tetrodes(self, n_tet):
        Parameters.num_tetrodes = n_tet

    def get_movement_ch(self):
        return self.movement_ch

    def set_movement_ch(self, movement_ch):
        Parameters.movement_ch = movement_ch

    def get_opto_ch(self):
        return self.opto_ch

    def set_opto_ch(self, opto_ch):
        Parameters.opto_ch = opto_ch

    def get_waveform_size(self):
        return self.waveform_size

    def set_waveform_size(self, waveform_size):
        Parameters.waveform_size = waveform_size

    def set_sampling_rate(self, sr):
        Parameters.sampling_rate = sr

    def get_sampling_rate(self):
        return self.sampling_rate

