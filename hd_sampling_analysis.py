import os
import matplotlib.pylab as plt
import power_spectra


def plot_power_spectrum_for_hd(freqs, idx, ps, prm):
    save_path = prm.get_local_recording_folder_path() + prm.get_sorter_name() + '/Figures/hd_power_spectrum'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.xlim(0, 15)
    plt.xlabel('Frequencies [Hz]')
    plt.plot(freqs[idx], ps[idx])
    plt.savefig(save_path + '/' + 'hd_power_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def check_if_hd_sampling_was_high_enough(spatial_data, params):
    freqs, idx, ps = power_spectra.power_spectrum(spatial_data.hd, params)
    #plot_power_spectrum_for_hd(freqs, idx, ps, params)