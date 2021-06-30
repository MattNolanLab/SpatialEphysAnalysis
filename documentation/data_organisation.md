## Data structures in open field analyses

To facilitate efficient and reproducible analysis of electrophysiological and behavioural data we have developed a framework that uses data frames implemented in Python. Our overall philosophy is to clearly separate pre-processing of data from analyses that directly address experimental questions. Bonsai, OpenEphys, and MountainSort are used to pre-process the experimental data. For example, Mountainsort is first used for spike clustering. The outputs of this pre-processing are used to initialize data frames that are then used for subsequent analyses.

The framework use two data frames, one for analyses at the level of behavioural sessions, and one for analyses at the level of spike clusters. Data from new behavioural sessions, or clusters, are added to the frames as new rows. Outputs of new analyses are added as new columns. Because the data frames store data and analyses, for multiple sessions and clusters respectively, it's straightforward to implement new analyses over many sessions or clusters without writing complicated looping code. The framework is currently implemented only for open field sessions. 

New analysis code should be added in a way that uses the data frames. If analyses require access to raw data, then a processing step should be used to add the required data to the data frames. Results of subsequent analyses should be added into these data frames as new columns. For instance, if we implement calculating the grid score of cells, this should be a new column in the data frame that contains information on clusters.

At the end of the analysis, three data frames are saved: the position data frame, the cluster data frame, and a thrird data frame that contains the rejected 'noisy' clusters. These are saved as pkl files in each recording folder in /DataFrames on the server.

## Description of two main data frames
The 'session' data frame contains processed data describing the position and head-direction of the animal. Each row is data from one session. The columns are organized as follows:

**synced_spatial_data** 
_(this is the name of the df in the main code)_

* synced_time : arrays of time in seconds, synchronized with the ephys data
* position_x : arrays of x coordinates of position of animal in arena in cm
* position_y : y coordinate of position of animal in arena in cm
* position_x_pixels : x coordinate of position of animal in arena in pixels
* position_y_pixels : y coordinate of position of animal in arena in pixels
* hd : head-direction of animal in degrees [-180;180]
* speed : speed of animal

`synced_spatial_data.head()`
![image](https://user-images.githubusercontent.com/16649631/43079289-9a13ab22-8e84-11e8-9b57-80518fdfda63.png)

***

The 'clusters' data frame contains data for each cluster and their spatial firing. Each row is a cluster. The columns are organized as follows:

**spatial_firing**
_(this is the name of the df in the main code)_

* session_id : name of main recording folder (example: M5_2018-03-06_15-34-44_of)
* cluster_id : id of cluster within session (1 - number of clusters)
* tetrode : id of tetrode within session (1 - 4)
* primary_channel : channel where the event was detected on (1 - 4)
* firing_times : array of all firing event times that belong to cluster from the open field exploration (in sampling points)
* number_of_spikes : total number of spikes in session excluding opto tagging part
* mean_firing_rate : total number of spikes / total time exclding opto tagging data [Hz]
* recording_length_seconds : length of recording (sec)
* isolation : cluster quality metric (see MountainSort paper)
* noise_overlap : cluster quality metric (see MountainSort paper)
* peak_snr : cluster quality metric (see MountainSort paper)
* peak_amp : amplitude of spike at peak (mV)
* all_snippets : all detected spike waveforms corresponding to cell
* firing_times_opto : array of firing events from the opto tagging part of the recording (in sampling points)
* position_x : x coordinate of position of animal in arena in cm corresponding to each firing event from the exploration
* position_y : y coordinate of position of animal in arena in cm corresponding to each firing event from the exploration
* position_x_pixels : x coordinate of position of animal in arena in pixels corresponding to each firing event from the exploration
* position_y_pixels : y coordinate of position of animal in arena in pixels corresponding to each firing event from the exploration
* hd : head-direction of animal in degrees corresponding to each firing event from the exploration [-180;180]
* speed : speed of animal corresponding to each firing event
* speed_score : The speed score is a measure of the correlation between the firing rate of the neuron and the running speed of the
animal. The firing times of the neuron are binned at the same sampling rate as the position data (speed). The resulting
temporal firing histogram is then smoothed with a Gaussian (standard deviation ~250ms). Speed and temporal firing rate
are correlated (Pearson correlation) to obtain the speed score.
Based on: Gois & Tort, 2018, Cell Reports 25, 1872–1884
* speed_score_p_values : p value corrsponding to speed score
* border_score : border scores according to Solstad et al (2008)
* corner_score : Corner scores and cue scores are also formalised loosely following the b = (cM - dm) / (cM + dm) structure.
* ThetaPower :
* ThetaIndex :
* Boccara_theta_class :
* firing_maps : binned data array for each cluster with firing rate maps
* rate_map_autocorrelogram : autocorrelogram of diring rate map (shifted in 2d)
* grid_spacing : Defined by Hafting, Fyhn, Molden, Moser, Moser (2005) as the distance from the central autocorrelogram peak to the
vertices of the inner hexagon in the autocorrelogram (the median of the six distances). This should be in cm.
* field_size : Defined by Wills, Barry, Cacucci (2012) as the square root of the area of the central peak of the autocorrelogram
divided by pi. (This should be in cm2.)
* grid_score : Defined by Krupic, Bauza, Burton, Barry, O'Keefe (2015) as the difference between the minimum correlation coefficient
for autocorrelogram rotations of 60 and 120 degrees and the maximum correlation coefficient for autocorrelogram
rotations of 30, 90 and 150 degrees. This score can vary between -2 and 2, although generally values above
below -1.5 or above 1.5 are uncommon.
* hd_spike_histogram : polar histogram of HD when the cell fired. For each degree the number of events are counted and then smoothing is done on this data by adding the values up in a (23 degree) rolling window. For each degree between 0 and 360 the number of events between n-window and n+window is added up. This histogram is then divided by the histogram obtained from all the HD data from the session divided by the sampling rate.
`spike_histogram = spike_histogram_hd/(hd_histogram_session/ephys_sampling_rate)`

This is then normalized on the plot hd_hist*(max(hd_hist_cluster))/max(hd_hist)) is plotted. 

_We should discuss whether this is a good way, it does not make a lot of sense to me. It is not exactly as in the MATLAB version._
* firing_fields : lists of indices that belong to an individual firing field detected. One cluster can have multiple lists. (Indices correspond to the firing rate map.)
For example on this image, the yellow circles represent the local maximum that the algorithm found and then all the blue parts around them were taken for that particular firing field. This cluster will have four lists added to its firing fields.

* firing_fields_hd_session : head-direction histograms that correspond to firing fields (each cluster has a list) - so this data is only from when the animal was in the given field
* firing_fields_hd_cluster : head-direction histograms that correspond to firing fields when the cell fired - this data is from when the animal was in the field AND the cell fired

* field_hd_p : Kuiper p values corresponding to the head-direction histograms of each field
* field_stat : Kuiper raw statistic corresponding to the head-direction histograms of each field
* field_hd_max_rate : maximum firing rate in field
* field_preferred_hd : preferred head-direction in field
* field_hd_score : hd score in field (see hd score definition above)
* field_max_firing_rate : max firing rate in given field among rate bins
* number_of_spikes_in_fields : number of spikes/firing events that occured in field
* time_spent_in_fields_sampling_points : amount of time the animal spent in the field
* spike_times_in_fields : specific firing times in field
* times_in_session_fields : time points when the animal was in the field

![image](https://user-images.githubusercontent.com/16649631/43839928-480eb3c2-9b17-11e8-96a4-f2da8b4de1c6.png)

* max_firing_rate : the highest among the firing rates of the bins of the rate map (Hz)

* max_firing_rate_hd : the highest among the firing rates of angles of the polar histogram (Hz)

* preferred_HD : the head-direction angle where the cell fires the most (highest rate), degrees

* hd_score : score between 0 and 1. The higher the score, the more head-direction specific the cell is.

`        dy = np.sin(angles_rad)`
        `dx = np.cos(angles_rad)`

        `totx = sum(dx * hd_hist)/sum(hd_hist)`
        `toty = sum(dy * hd_hist)/sum(hd_hist)`
        `r = np.sqrt(totx*totx + toty*toty)`
        `hd_scores.append(r)`

* hd_p : result of two-sample Kuiper test on the distribution of hd from the whole session and the distribution of hd when the cell fired. The probability of obtaining two samples this different from the same distribution.

* hd_stat : the raw test statistic from the Kuiper test described above
* rayleigh_score :  This test is  used to identify a non-uniform distribution, i.e. it is designed for detecting an unimodal deviation from uniformity. More precisely, it assumes the following hypotheses:
    - H0 (null hypothesis): The population is distributed uniformly around the
    circle.
    - H1 (alternative hypothesis): The population is not distributed uniformly
    around the circle.
    Small p-values suggest to reject the null hypothesis

### Circular statistics are done in R using the circular.r package 

(documentation: https://cran.r-project.org/web/packages/circular/circular.pdf)

* watson_test_hd - stats restuls of two sample Watson test comparing distribution of HD from the whole session to the HD when the cell fired. p value ranged can be inferred from stats
https://github.com/MattNolanLab/in_vivo_ephys_openephys/blob/add_python_post_clustering/PostSorting/process_fields.r

* kuiper_cluster - one sample Kuiper test stats for HD when the cell fired

* kuiper_session - one sample Kuiper test stats for HD from the whole session

* watson_cluster - one sample Watson test stats for HD when the cell fired

* watson_session - one sample Watson test stats for HD for the whole session


* field_corr_r : correlation betwen head direction in a field from the first and second half of the session
* field_corr_p : p value coeesponding to field_corr_r
* hd_correlation_first_vs_second_half : correlation between head direction between the first and second half of the session (from the entire recordings)
* hd_correlation_first_vs_second_half_p : corresponds to r above
* hd_hist_first_half : 'classic' hd histogram from first hald of session
* hd_hist_second_half : 'classic' hd histogram from second half of session
* rate_map_correlation_first_vs_second_half : correlatiin between rate maps made from the data from the first vs the second half of the session
* percent_excluded_bins_rate_map_correlation_first_vs_second_half_p : percentage of bins excluded from the rate map correlation analysis

* spike_times_after_opto : firing times after opto pulses
* opto_latencies : list of latncies of spikes in the first 10ms after opto pulses
* opto_latencies_mean_ms : avg latencies after opto pulses
* opto_latencies_sd_ms : standard dev of latencies
* random_snippets_opto : random snippet waveforms from the opto-tagging part of the recording
* random_first_spike_snippets_opto : random snippet waveforms of spikes that happened immediately after opto pulses
* SALT_p : SALT test p value (Kvitsiani D*, Ranade S*, Hangya B, Taniguchi H, Huang JZ, Kepecs A (2013) Distinct behavioural and network correlates of two interneuron types in prefrontal cortex. Nature 498:363?6.)
* SALT_I : SALT test test tatistic
* SALT_latencies : latencies tested in SALT test


## Data structures in virtual reality recordings
### parameters 

Specific parameters need to be set for the vr analysis environment. 


* **stop_threshold** this is the value in which the animals speed has to drop below for a stop to be extracted (<0.7 cm/second)
* **movement_channel** this is the pin on the DAQ which has the movement of the animal along the track
* **first_trial_channel** this is the first pin on the DAQ which has the trial type information 
* **second_trial_channel** this is the first pin on the DAQ which has the trial type information 


## Structure of dataframes

The spatial data frame contains processed data describing the position of the animal in the virtual reality. The columns are organized as follows:

**vr_spatial_data (name of the df in the main code)**

* time_ms : arrays of time in seconds, synchronized with the ephys data
* position_cm : arrays of x coordinates of position of animal in virtual track in cm, synchronized with the ephys data
* trial_number : arrays of the current trial number, synchronized with the ephys data
* trial_type : arrays of the current trial type (beaconed, non beaconed, probe), synchronized with the ephys data
* velocity : instant velocity of animal (cm/s), synchronized with the ephys data
* speed : speed of animal averaged over 200 ms (cm/s), synchronized with the ephys data
* stops : whether an animal has stopped (0/1 : no/yes), synchronized with the ephys data
* filtered_stops : stops within 1 cm of each other are removed 
* stop_times : array of times which the animal has stopped


**spike_data**
_(this is the name of the df in the main code)_

* session_id : name of main recording folder (example: M5_2018-03-06_15-34-44_of)
* cluster_id : id of cluster within session (1 - number of clusters)
* tetrode : id of tetrode within session (1 - 4)
* primary_channel : channel where the event was detected on (1 - 4)
* firing_times : array of all firing event times that belong to cluster from the vr (in sampling points)
