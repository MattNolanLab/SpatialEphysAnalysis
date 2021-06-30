import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis
import PostSorting.vr_extract_data
import PostSorting.vr_cued_make_plots
import PostSorting.vr_spatial_data
from numpy import inf
import gc
import matplotlib.ticker as ticker
import pandas as pd
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import settings
'''

# Plot basic info to check recording is good:
> movement channel
> trial channels (one and two)

'''

# plot the raw movement channel to check all is good
def plot_movement_channel(location, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(location)
    plt.savefig(save_path + '/movement' + '.png')
    plt.close()

# plot the trials to check all is good
def plot_trials(trials, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(trials)
    plt.savefig(save_path + '/trials' + '.png')
    plt.close()

def plot_velocity(velocity, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/velocity' + '.png')
    plt.close()

def plot_running_mean_velocity(velocity, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/running_mean_velocity' + '.png')
    plt.close()

# plot the raw trial channels to check all is good
def plot_trial_channels(trial1, trial2, output_path):
    plt.plot(trial1[0,:])
    plt.savefig(output_path + '/Figures/trial_type1.png')
    plt.close()
    plt.plot(trial2[0,:])
    plt.savefig(output_path + '/Figures/trial_type2.png')
    plt.close()


'''

# Plot behavioural info:
> stops on trials 
> avg stop histogram
> avg speed histogram
> combined plot

'''

def get_trial_color(trial_type):
    if trial_type == 0:
        return "black"
    elif trial_type == 1:
        return "red"
    elif trial_type == 2:
        return "blue"
    else:
        print("invalid trial-type passed to get_trial_color()")

def plot_stops_on_track(processed_position_data, output_path, track_length=200):
    print('I am plotting stop rasta...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)

        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color=trial_stop_color, markersize=4)

    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()

def min_max_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


def plot_firing_rate_maps_per_trial(spike_data, prm):
    print('plotting trial firing rate maps...')
    save_path = prm.get_output_path() + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:

            x_max = max(np.array(spike_data.beaconed_trial_number.iloc[cluster_index])) + 1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_maps = np.array(spike_data["firing_rate_maps"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0

            if len(cluster_firing_maps) == 0:
                print("stop here")

            cluster_firing_maps = min_max_normalize(cluster_firing_maps)

            cmap = plt.cm.get_cmap("jet")
            cmap.set_bad(color='white')
            c = ax.imshow(cluster_firing_maps, interpolation='none', cmap=cmap, vmin=0, vmax=np.max(cluster_firing_maps), origin='lower')
            #clb = fig.colorbar(c, ax=ax, shrink=0.8)
            #clb.set_clim(0, max_pwr_shown)
            #clb.set_label(label='Power',size=20)
            #clb.set_ticks([0, max_power])
            #clb.set_ticklabels(["0", r'$\geq$'+str(max_power)])
            #clb.ax.tick_params(labelsize=15)

            #for i in range(len(cluster_firing_maps)):
            #    for j in range(len(cluster_firing_maps[0])):
            #        ax.scatter(j+0.5,i+0.5, marker="s", s=1, c=cluster_firing_maps[i,j], cmap=cm.jet)

            plt.ylabel('Trial Number', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            #plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '.png', dpi=200)
            plt.close()


def plot_stop_histogram(processed_position_data, output_path, track_length=200):
    print('plotting stop histogram...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    bin_size = 5

    beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 1]
    probe_trials = processed_position_data[processed_position_data["trial_type"] == 2]

    beaconed_stops = plot_utility.pandas_collumn_to_numpy_array(beaconed_trials["stop_location_cm"])
    non_beaconed_stops = plot_utility.pandas_collumn_to_numpy_array(non_beaconed_trials["stop_location_cm"])
    probe_stops = plot_utility.pandas_collumn_to_numpy_array(probe_trials["stop_location_cm"])

    beaconed_stop_hist, bin_edges = np.histogram(beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    non_beaconed_stop_hist, bin_edges = np.histogram(non_beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    probe_stop_hist, bin_edges = np.histogram(probe_stops, bins=int(track_length/bin_size), range=(0, track_length))
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    ax.plot(bin_centres, beaconed_stop_hist/len(beaconed_trials), '-', color='Black')
    ax.plot(bin_centres, non_beaconed_stop_hist/len(non_beaconed_trials), '-', color='Red')
    if len(probe_trials)>0:
        ax.plot(bin_centres, probe_stop_hist/len(probe_trials), '-', color='Blue')

    plt.ylabel('Stops/Trial', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)

    maxes = [max(beaconed_stop_hist/len(beaconed_trials)), max(non_beaconed_stop_hist/len(non_beaconed_trials))]
    x_max = max(maxes)+(0.1*max(maxes))
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
    plt.close()

def plot_speed_per_trial(processed_position_data, output_path, track_length=200):
    print('plotting speed heatmap...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    x_max = len(processed_position_data)
    if x_max>100:
        fig = plt.figure(figsize=(4,(x_max/32)))
    else:
        fig = plt.figure(figsize=(4,(x_max/20)))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    trial_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned"])
    cmap = plt.cm.get_cmap("jet")
    cmap.set_bad(color='white')
    trial_speeds = np.clip(trial_speeds, a_min=0, a_max=60)
    c = ax.imshow(trial_speeds, interpolation='none', cmap=cmap, vmin=0, vmax=60)
    clb = fig.colorbar(c, ax=ax, shrink=0.5)
    clb.mappable.set_clim(0, 60)
    plt.ylabel('Trial Number', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/speed_heat_map' + '.png', dpi=200)
    plt.close()

def plot_speed_histogram(processed_position_data, output_path, track_length=200):
    trial_averaged_beaconed_speeds, trial_averaged_non_beaconed_speeds, trial_averaged_probe_speeds = \
        PostSorting.vr_spatial_data.trial_average_speed(processed_position_data)

    print('plotting speed histogram...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    speed_histogram = plt.figure(figsize=(6,4))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])

    if len(trial_averaged_beaconed_speeds)>0:
        ax.plot(bin_centres, trial_averaged_beaconed_speeds, '-', color='Black')

    if len(trial_averaged_non_beaconed_speeds)>0:
        ax.plot(bin_centres, trial_averaged_non_beaconed_speeds, '-', color='Red')

    if len(trial_averaged_probe_speeds)>0:
        ax.plot(bin_centres, trial_averaged_probe_speeds, '-', color='Blue')

    plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)

    max_b = max(trial_averaged_beaconed_speeds)
    max_nb = max(trial_averaged_non_beaconed_speeds)
    if len(trial_averaged_probe_speeds)>0:
        max_p = max(trial_averaged_probe_speeds)
        x_max = max([max_b, max_nb, max_p])
    else:
        x_max = max([max_b, max_nb])

    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/speed_histogram' + '.png', dpi=200)
    plt.close()


def plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=200,
                         plot_trials=["beaconed", "non_beaconed", "probe"]):
    print('plotting spike rastas...')
    save_path = output_path + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        if len(firing_times_cluster)>1:

            x_max = len(processed_position_data)+1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            if "beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].beaconed_position_cm, cluster_spike_data.iloc[0].beaconed_trial_number, '|', color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].nonbeaconed_position_cm, cluster_spike_data.iloc[0].nonbeaconed_trial_number, '|', color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].probe_position_cm, cluster_spike_data.iloc[0].probe_trial_number, '|', color='Blue', markersize=4)

            plt.ylabel('Spikes on trials', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            plot_utility.style_track_plot(ax, track_length)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()


def plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=200):
    gauss_kernel = Gaussian1DKernel(2)
    print('I am plotting firing rate maps...')
    save_path = output_path + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        avg_beaconed_spike_rate = np.array(cluster_spike_data["beaconed_firing_rate_map"].to_list()[0])
        avg_nonbeaconed_spike_rate = np.array(cluster_spike_data["non_beaconed_firing_rate_map"].to_list()[0])
        avg_probe_spike_rate = np.array(cluster_spike_data["probe_firing_rate_map"].to_list()[0])

        beaconed_firing_rate_map_sem = np.array(cluster_spike_data["beaconed_firing_rate_map_sem"].to_list()[0])
        non_beaconed_firing_rate_map_sem = np.array(cluster_spike_data["non_beaconed_firing_rate_map_sem"].to_list()[0])
        probe_firing_rate_map_sem = np.array(cluster_spike_data["probe_firing_rate_map_sem"].to_list()[0])

        avg_beaconed_spike_rate = convolve(avg_beaconed_spike_rate, gauss_kernel) # convolve and smooth beaconed
        beaconed_firing_rate_map_sem = convolve(beaconed_firing_rate_map_sem, gauss_kernel)

        avg_nonbeaconed_spike_rate = convolve(avg_nonbeaconed_spike_rate, gauss_kernel) # convolve and smooth non beaconed
        non_beaconed_firing_rate_map_sem = convolve(non_beaconed_firing_rate_map_sem, gauss_kernel)

        if len(avg_probe_spike_rate)>0:
            avg_probe_spike_rate = convolve(avg_probe_spike_rate, gauss_kernel) # convolve and smooth probe
            probe_firing_rate_map_sem = convolve(probe_firing_rate_map_sem, gauss_kernel)

        avg_spikes_on_track = plt.figure(figsize=(6,4))
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])

        #plotting the rates are filling with the standard error around the mean
        ax.plot(bin_centres, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bin_centres, avg_beaconed_spike_rate-beaconed_firing_rate_map_sem,
                                     avg_beaconed_spike_rate+beaconed_firing_rate_map_sem, color="Black", alpha=0.5)

        ax.plot(bin_centres, avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.fill_between(bin_centres, avg_nonbeaconed_spike_rate-non_beaconed_firing_rate_map_sem,
                                     avg_nonbeaconed_spike_rate+non_beaconed_firing_rate_map_sem, color="Red", alpha=0.5)

        if len(avg_probe_spike_rate)>0:
            ax.plot(bin_centres, avg_probe_spike_rate, '-', color='Blue')
            ax.fill_between(bin_centres, avg_probe_spike_rate-probe_firing_rate_map_sem,
                                         avg_probe_spike_rate+probe_firing_rate_map_sem, color="Blue", alpha=0.5)

        #plotting jargon
        if track_length == 200:
            ax.locator_params(axis = 'x', nbins=3)
            ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,track_length)
        nb_x_max = np.nanmax(avg_beaconed_spike_rate)
        b_x_max = np.nanmax(avg_nonbeaconed_spike_rate)
        if b_x_max > nb_x_max:
            plot_utility.style_vr_plot(ax, b_x_max)
        elif b_x_max < nb_x_max:
            plot_utility.style_vr_plot(ax, nb_x_max)
        plot_utility.style_track_plot(ax, track_length)
        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.12, right=0.87, top=0.92)

        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_rate_map_Cluster_' + str(cluster_id) + '.png', dpi=200)
        plt.close()

'''
plot gaussian convolved firing rate in time against similarly convolved speed and location. 
'''

def plot_convolved_rates_in_time(spike_data, prm):
    print('plotting spike rastas...')
    save_path = prm.get_output_path() + '/Figures/ConvolvedRates_InTime'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure(figsize=(4,5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        firing_rate = spike_data.loc[cluster_index].spike_rate_in_time
        speed = spike_data.loc[cluster_index].speed_rate_in_time
        x_max= np.max(firing_rate)
        ax.plot(firing_rate, speed, '|', color='Black', markersize=4)
        plt.ylabel('Firing rate (Hz)', fontsize=12, labelpad = 10)
        plt.xlabel('Speed (cm/sec)', fontsize=12, labelpad = 10)
        #plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #plot_utility.style_track_plot(ax, 200)
        #plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_versus_SPEED_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

        spikes_on_track = plt.figure(figsize=(4,5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        position = spike_data.loc[cluster_index].position_rate_in_time
        ax.plot(firing_rate, position, '|', color='Black', markersize=4)
        plt.ylabel('Firing rate (Hz)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        # ]polt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #plot_utility.style_track_plot(ax, 200)
        #plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_versus_POSITION_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

def make_plots(processed_position_data, spike_data, output_path, track_length=settings.track_length):
    # Create plots for the VR experiments
    
    plot_stops_on_track(processed_position_data, output_path, track_length=track_length)
    plot_stop_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_per_trial(processed_position_data, output_path, track_length=track_length)

    if spike_data is not None:
        PostSorting.make_plots.plot_waveforms(spike_data, output_path)
        PostSorting.make_plots.plot_spike_histogram(spike_data, output_path)
        PostSorting.make_plots.plot_autocorrelograms(spike_data, output_path)
        gc.collect()
        plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=track_length)
        plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=track_length,
                             plot_trials=["beaconed", "non_beaconed", "probe"])


def plot_field_centre_of_mass_on_track(spike_data, prm, plot_trials=["beaconed", "non_beaconed", "probe"]):

    print('plotting field rastas...')
    save_path = prm.get_output_path() + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:

            x_max = max(np.array(spike_data.beaconed_trial_number.iloc[cluster_index])) + 1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_trial_numbers = np.array(spike_data["fields_com_trial_number"].iloc[cluster_index])
            cluster_firing_com_trial_types = np.array(spike_data["fields_com_trial_type"].iloc[cluster_index])

            if "beaconed" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 0], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 0], "s", color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 1], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 1], "s", color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 2], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 2], "s", color='Blue', markersize=4)

            #ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=3)
            plt.ylabel('Field COM on trials', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_number_of_fields(cluster_df, processed_position_data, prm):
    print('plotting field rastas...')
    save_path = prm.get_output_path() + '/Figures/field_analysis'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    cluster_id = cluster_df["cluster_id"].iloc[0]

    cluster_firing_com = np.array(cluster_df["fields_com"].iloc[0])
    cluster_firing_com_trial_numbers = np.array(cluster_df["fields_com_trial_number"].iloc[0])
    cluster_firing_com_trial_types = np.array(cluster_df["fields_com_trial_type"].iloc[0])

    beaconed_firing_com = cluster_firing_com[cluster_firing_com_trial_types == 0]
    non_beaconed_firing_com = cluster_firing_com[cluster_firing_com_trial_types == 1]
    probe_firing_com = cluster_firing_com[cluster_firing_com_trial_types == 2]

    n_beaconed_trials = processed_position_data.beaconed_total_trial_number.iloc[0]
    n_nonbeaconed_trials = processed_position_data.nonbeaconed_total_trial_number.iloc[0]
    n_probe_trials = processed_position_data.probe_total_trial_number.iloc[0]

def plot_inter_field_distance_histogram(spike_data, prm):
    print('plotting field com histogram...')
    bin_size=5
    tick_spacing = 50
    save_path = prm.get_output_path() + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_com_distances = np.array(spike_data["distance_between_fields"].iloc[cluster_index])

            field_hist, bin_edges = np.histogram(cluster_firing_com_distances, bins=int(prm.get_track_length()/bin_size), range=[0, prm.get_track_length()], density=True)

            ax.bar(bin_edges[:-1], field_hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
            plt.ylabel('Field Density', fontsize=12, labelpad = 10)
            plt.xlabel('Field to Field Distance (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            x_max = max(field_hist)
            #plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_distance_hist_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()


def plot_field_com_histogram(spike_data, prm):
    bin_size = 5
    tick_spacing = 50

    print('plotting field com histogram...')
    save_path = prm.get_output_path() + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])

            field_hist, bin_edges = np.histogram(cluster_firing_com, bins=int(prm.get_track_length()/bin_size), range=[0, prm.get_track_length()], density=True)

            ax.bar(bin_edges[:-1], field_hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
            plt.ylabel('Field Density', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            field_hist = np.nan_to_num(field_hist)

            x_max = max(field_hist)
            plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_hist_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()


def plot_field_analysis(spike_data, processed_position_data, prm):

    print('plotting field rastas...')
    save_path = prm.get_output_path() + '/Figures/field_analysis'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_times_cluster = cluster_df.firing_times.iloc[0]

        if len(firing_times_cluster)>1:
            plot_number_of_fields(cluster_df, processed_position_data, prm)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_trial_numbers = np.array(spike_data["fields_com_trial_number"].iloc[cluster_index])
            cluster_firing_com_trial_types = np.array(spike_data["fields_com_trial_type"].iloc[cluster_index])

            beaconed_firing_com = cluster_firing_com[cluster_firing_com_trial_types == 0]
            non_beaconed_firing_com = cluster_firing_com[cluster_firing_com_trial_types == 0]
            probe_firing_com = cluster_firing_com[cluster_firing_com_trial_types == 0]

            plt.ylabel('Field COM on trials', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()





