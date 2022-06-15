# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:30:48 2018.

@author: imbroscb
"""

import numpy as np


def get_format_data(data, ch, first_sweep, first_point=None, last_point=None):
    """
    Get the channels to use for analysis.

    This function returns the channels to use for analysis, concatenates the
    sweeps (if there are more sweeps) and removes the initial or final part of
    each sweep if necessary.
    """
    output_data = {}
    for key, sweeps in data.items():
        if key in ch:
            sweeps_to_analyse = []
            for sw in range(len(sweeps)):
                if sw >= first_sweep:
                    sweeps_to_analyse.append(
                        sweeps[sw][first_point:last_point, :])
            output_data[int(key[-1])] = np.asarray(
                sweeps_to_analyse, dtype=('object')).reshape(-1, 1)

    return output_data
