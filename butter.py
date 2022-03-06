# -*- coding: utf-8 -*-

from scipy import signal


def lowpass(data, cutoff_frequency, fs=20000, order=2):
    """
   Created on Wed Sep  5 17:13:00 2018

    lowpass filter the input signal

    inputs:
        data = the signal to be filtered. It needs to have shape [-1, 1]
        cutoff_frequency = the cutoff frequency
        fs = the sampling rate (Hz)
        order = polynomial
    output:
        the filtered_data

    @author: imbroscb
    """
    nyq = fs * 0.5
    b, a = signal.butter(order, cutoff_frequency / nyq)
    if len(data.shape) > 1:
        if data.shape[0] > data.shape[1]:
            ref_filt = signal.lfilter(b, a, data, axis=0)
        else:
            ref_filt = signal.lfilter(b, a, data, axis=1)
    else:
        ref_filt = signal.lfilter(b, a, data, axis=0)
    return ref_filt


def bandpass(data, cutoff_low, cutoff_high, fs=20000, order=2):
    """
    Created on Wed Sep  5 17:13:00 2018

    bandpass filter the input signal
    inputs:
        data = the signal to be filtered. It needs to have shape [-1, 1]
        cutoff_low = the low cutoff frequency
        cutoff_high = the high cutoff frequency
        fs = the sampling rate (Hz)
        order = polynomial
    output:
        the filtered_data

    @author: imbroscb
    """
    nyq = fs * 0.5
    b, a = signal.butter(order, [cutoff_low / nyq, cutoff_high / nyq],
                         btype='band')
    if len(data.shape) > 1:
        if data.shape[0] > data.shape[1]:
            ref_filt = signal.lfilter(b, a, data, axis=0)
        else:
            ref_filt = signal.lfilter(b, a, data, axis=1)
    else:
        ref_filt = signal.lfilter(b, a, data, axis=0)
    return ref_filt


def highpass(data, cutoff_frequency, fs=20000, order=2):
    """
    Created on Wed Sep  5 17:13:00 2018

    highpass filter the input signal
    inputs:
        data = the signal to be filtered. It needs to have shape [-1,1]
        cutoff_frequency = the cutoff frequency
        fs = the sampling rate (Hz)
        order = polynomial
    output:
        the filtered_data

    @author: imbroscb
    """
    nyq = fs * 0.5
    b, a = signal.butter(order, cutoff_frequency / nyq, btype='highpass')
    if len(data.shape) > 1:
        if data.shape[0] > data.shape[1]:
            ref_filt = signal.lfilter(b, a, data, axis=0)
        else:
            ref_filt = signal.lfilter(b, a, data, axis=1)
    else:
        ref_filt = signal.lfilter(b, a, data, axis=0)
    return ref_filt
