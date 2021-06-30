from __future__ import division
import numpy as np


def power_spectrum(data, prm):
    # data = np.random.rand(301) - 0.5
    ps = np.abs(np.fft.fft(data))**2

    time_step = 1 / 30  # 30 Hz sampling
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    return freqs, idx, ps

