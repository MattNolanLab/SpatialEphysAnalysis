import numpy as np


def make_baseline_latency_histogram(baseline_trials, bins, windows, binsize, nwins):
    nbins = len(bins)   # number of bins for latency histograms
    hlsi = np.zeros((nbins - 1, nwins))   # preallocate latency histograms
    nhlsi = np.zeros((nbins - 1, nwins))    # preallocate latency histograms
    for i in range(nwins - 1):   # loop through baseline windows
        min_spike_times = []
        for j, trial in enumerate(baseline_trials):   # loop through trials
            mask = (trial < windows[i + 1]) & (trial > windows[i])
            spikes_in_win = np.array(trial)[mask]
            if len(spikes_in_win) > 0:
                min_spike_times.append(spikes_in_win.min() - windows[i])   # latency from window
            else:
                min_spike_times.append(- binsize / 2)   # 0 if no spike in the window
        hlsi[:, i], _ = np.histogram(min_spike_times, bins)   # latency histogram
        nhlsi[:, i] = hlsi[:, i] / sum(hlsi[:, i])   # normalized latency histogram
    return hlsi, nhlsi


def get_js_divergence(kn, nhlsi):
    jsd = np.zeros((kn, kn)) * np.nan
    for k1 in range(kn):
        D1 = nhlsi[:, k1]  # 1st latency histogram
        for k2 in range(k1 + 1, kn):
            D2 = nhlsi[:, k2]  # 2nd latency histogram
            jsd[k1, k2] = np.sqrt(JSdiv(D1, D2) * 2)  # pairwise modified JS-divergence (real metric!)
    return jsd


def get_js_divergence_for_latency(test_trials, latency, latency_hist, latency_hist_normalized, window_size,
                                  bin_size, number_of_windows, bins):
    min_spike_times = []
    for trial_index, trial in enumerate(test_trials):  # loop through trials
        mask = (trial < latency + window_size) & (trial > latency)
        spikes_in_win = np.array(trial)[mask]
        if len(spikes_in_win) > 0:
            min_spike_times.append(spikes_in_win.min() - latency)  # latency from window
        else:
            min_spike_times.append(- bin_size / 2)  # 0 if no spike in the window
    latency_hist[:, number_of_windows - 1], _ = np.histogram(min_spike_times, bins)  # latency histogram
    latency_hist_normalized[:, number_of_windows - 1] = latency_hist[:, number_of_windows - 1] / sum(
        latency_hist[:, number_of_windows - 1])  # normalized latency histogram
    # JS-divergence
    jsd = get_js_divergence(number_of_windows, latency_hist_normalized)
    return jsd


def salt(baseline_trials, test_trials, window_size=0.01, latency_step=0.01, baseline_start=0, baseline_end=0.02,
         test_start=0, test_end=0.02):
    '''SALT   Stimulus-associated spike latency test.
    Calculates a modified version of Jensen-Shannon divergence (see [1]_)
    for spike latency histograms. Please cite [2]_ when using this program.

    Parameters
    ----------
    baseline_trials (seconds) : Spike raster for stimulus-free baseline
       period. The baseline period has to excede the window size (winsize)
       multiple times, as the length of the baseline segment divided by the
       window size determines the sample size of the null
       distribution (see below).
    test_trials (seconds): Spike raster for test period, i.e. after
       stimulus. The test period has to excede the window size (winsize)
       multiple times, as the length of the test period divided by the
       latency_step size determines the number of latencies to be tested.
    window_size (seconds) : Window size for baseline and test windows in seconds
        (optional default, 0.01 s).
    latency_step (seconds) : Step size for test latencies in seconds
        (optional default, 0.01 s).


    Returns
    -------
    latencies : list
        latencies tested
    p_values : list
        Resulting P values for the Stimulus-Associated spike Latency Test.
    I_values : list
        Test statistic, difference between within baseline and test-to-baseline
        information distance values.

    Notes
    -----
    Briefly, the baseline binned spike raster (baseline_trials) is cut to
    non-overlapping epochs (window size determined by WN) and spike latency
    histograms for first spikes are computed within each epoch. A similar
    histogram is constructed for the test epoch (test_trials). Pairwise
    information distance measures are calculated for the baseline
    histograms to form a null-hypothesis distribution of distances. The
    distances of the test histogram and all baseline histograms are
    calculated and the median of these values is tested against the
    null-hypothesis distribution, resulting in a p value (P).

    References
    ----------
    .. [1] res DM, Schindelin JE (2003) A new metric for probability
       distributions. IEEE Transactions on Information Theory 49:1858-1860.

    .. [2] Kvitsiani D*, Ranade S*, Hangya B, Taniguchi H, Huang JZ, Kepecs A
       (2013) Distinct behavioural and network correlates of two interneuron
       types in prefrontal cortex. Nature 498:363?6.'''
    windows = np.arange(baseline_start, baseline_end + window_size, window_size)
    bin_size = window_size / 20
    bins = np.arange(- bin_size, window_size + bin_size, bin_size)
    number_of_windows = len(windows)
    # Latency histogram - baseline
    latency_hist, latency_hist_normalized = make_baseline_latency_histogram(baseline_trials, bins, windows, bin_size, number_of_windows)
    latencies = np.arange(0, test_end + latency_step, latency_step)
    p_values = []
    I_values = []
    for latency in latencies:
        jsd = get_js_divergence_for_latency(test_trials, latency, latency_hist, latency_hist_normalized, window_size, bin_size, number_of_windows, bins)
        # Calculate p-value and information difference
        p, I = calculate_p_value(jsd, number_of_windows)
        p_values.append(p)
        I_values.append(I)
    return latencies, p_values, I_values


def calculate_p_value(kld, kn):
    """
    Calculates p value from distance matrix.
    """

    pnhk = kld[:kn - 1, :kn - 1]
    nullhypkld = pnhk[np.isfinite(pnhk)]   # nullhypothesis
    testkld = np.median(kld[:kn - 1, kn - 1])  # value to test
    sno = len(nullhypkld)   # sample size for nullhyp. distribution
    p_value = sum(nullhypkld >= testkld) / sno
    Idiff = testkld - np.median(nullhypkld)   # information difference between baseline and test min_spike_times
    return p_value, Idiff


def JSdiv(P, Q):
    """JSDIV   Jensen-Shannon divergence.
    Calculates the Jensen-Shannon divergence of the two
    input distributions.
    """
    assert abs(sum(P)-1) < 0.00001 or abs(sum(Q)-1) < 0.00001,\
        'Input arguments must be probability distributions.'

    assert P.size == Q.size, 'Input distributions must be of the same size.'

    # JS-divergence
    M = (P + Q) / 2
    D1 = KLdist(P, M)
    D2 = KLdist(Q, M)
    D = (D1 + D2) / 2
    return D


def KLdist(P, Q):
    '''KLDIST   Kullbach-Leibler distance.
    Calculates the Kullbach-Leibler distance (information
    divergence) of the two input distributions.'''
    assert abs(sum(P)-1) < 0.00001 or abs(sum(Q)-1) < 0.00001,\
        'Input arguments must be probability distributions.'

    assert P.size == Q.size, 'Input distributions must be of the same size.'

    # KL-distance
    P2 = P[P * Q > 0]     # restrict to the common support
    Q2 = Q[P * Q > 0]
    P2 = P2 / sum(P2)  # renormalize
    Q2 = Q2 / sum(Q2)

    D = sum(P2 * np.log(P2 / Q2))
    return D


def run_salt_test_on_test_data():
    baseline_trials = [[0, 0.001], [0.002, 0.003, 0.005]]  # list of firing times for each trial (baseline)
    test_trials = [[0.19], [0.18]]  # list of times from test trials (after light)
    latencies, p_values, I_values = salt(baseline_trials=baseline_trials,
                                         test_trials=test_trials,
                                         window_size=0.01, baseline_start=0, baseline_end=0.02, test_start=0, test_end=0.02)
    # print(latencies)
    # print(p_values)


def turn_binary_array_to_time_series(binary_array, sampling_rate=30000):
    firing_times_list = []
    number_of_trials = binary_array.shape[0]
    for trial in range(number_of_trials):
        times_trial = []
        trial_data = binary_array[trial, :]
        spike_indices = np.where(trial_data == 1)
        for spike_index in spike_indices[0]:
            firing_time = (spike_index / sampling_rate)
            times_trial.append(firing_time)
        firing_times_list.append(times_trial)
    return firing_times_list


def convert_peristimulus_data_to_baseline_and_test(peristimulus_data):
    middle_of_window = int((peristimulus_data.shape[1] - 2) / 2)
    baseline_binary = peristimulus_data.values[:, 2:middle_of_window].astype(float)
    test_binary = peristimulus_data.values[:, middle_of_window:].astype(float)
    baseline = turn_binary_array_to_time_series(baseline_binary)
    test = turn_binary_array_to_time_series(test_binary)
    return baseline, test


def run_salt_test_on_peristimulus_data(spatial_firing, peristimulus_data):
    """
    This version of the SALT test slightly differs from the published MATLAB code and runs the test on multiple
    windows after the light pulse. This could be useful when looking at longer latency responses.
    :param spatial_firing: data frame where each row is a cell
    :param peristimulus_data: binary matrix where each row is a trial (light pulse) and 1s represent spikes and 0s
    no spikes
    :return: spatial_firing: data frame where each row is a cell - now containing SALT results
    """
    print('I will run the SALT test now.')
    all_latencies = []
    all_p_values = []
    all_i_values = []
    for cluster_index, cluster in spatial_firing.iterrows():
        # get relevant part of peristimulus data here
        peristim_cluster = peristimulus_data[peristimulus_data.cluster_id.astype(int) == cluster.cluster_id]
        baseline_trials, test_trials = convert_peristimulus_data_to_baseline_and_test(peristim_cluster)
        latencies, p_values, I_values = salt(baseline_trials=baseline_trials,
                                             test_trials=test_trials,
                                             window_size=0.01, latency_step=0.01, baseline_start=0, baseline_end=0.2, test_start=0, test_end=0.2)

        all_latencies.append(latencies)
        all_p_values.append(p_values)
        all_i_values.append(I_values)
    spatial_firing['SALT_p'] = all_p_values
    spatial_firing['SALT_I'] = all_i_values
    spatial_firing['SALT_latencies'] = all_latencies
    return spatial_firing

