# -*- coding: utf-8 -*-

import numpy as np


def get_channels(data, ch, ch_numb, start_at):

    """
    Created on Tue Sep 11 14:30:48 2018

    This function returns the channels to use for analysis, concatenates the
    sweeps and removes the initial part of each sweep if necessary.

    inputs:
        data = data organized in a dictionary obtained with the function
        load_abf
        position = a list with the names of the recording location
        fs = the sampling rate (Hz)
        start_at = the datapoint of each sweep from which to start the
                   concatenation, the datapoints before will be excluded

    output:
        data = the recordings in the right configuration to run the analysis
               organized as a dictionary with the names of the recording
               locations as key

    @author: imbroscb
            """


    output_data = {}
    c = 0
    for key, sweeps in data.items():
        if key in ch:
            for sw in range(len(sweeps)):
                sweeps[sw] = sweeps[sw][start_at:, :]
                output_data[ch_numb[c]] = np.asarray(
                    sweeps, dtype=('object')).reshape(-1, 1)
            c += 1

    return output_data
