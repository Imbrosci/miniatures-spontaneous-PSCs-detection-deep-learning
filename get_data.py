# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:30:48 2018.

@author: imbroscb
"""
import numpy as np


def get_format_data(data, ch, start_sweep, start_at):
    """
    Get the channels to use for analysis.

    This function returns the channels to use for analysis, concatenates the
    sweeps and removes the initial part of each sweep if necessary.

    inputs:
        data = data organized in a dictionary obtained with the function
        load_abf
        ch = list of channels name with the recordings
        start_at = the datapoint of each sweep from which to start the
                   concatenation, the datapoints before will be excluded

    output:
        data = the recordings in the right configuration to run the analysis
               organized as a dictionary with the names of the recording
               locations as key
    """
    output_data = {}
    for key, sweeps in data.items():
        if key in ch:
            sweeps_to_analyse = []
            for sw in range(len(sweeps)):
                if sw >= start_sweep:
                    sweeps_to_analyse.append(sweeps[sw][start_at:, :])
            output_data[int(key[-1])] = np.asarray(sweeps_to_analyse, dtype=('object')).reshape(-1, 1)

    return output_data
