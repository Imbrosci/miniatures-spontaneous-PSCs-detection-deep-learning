# -*- coding: utf-8 -*-

from neo import io
import numpy as np


def load_abf(filename):
    """
    Created on Tue Sep 11 10:52:17 2018

    extract the analog signal(s) from ABF files
    (generally acquired with clampex)

    @author: imbroscb
    """
    data = io.AxonIO(filename)
    bl = data.read()[0]
    data_out = {}
    ch_numb = 0
    # loop through the analogsignals
    for i in range(len(bl.segments[0].analogsignals)):
        # loop through the channels in each analogsignal
        for ch in range(bl.segments[0].analogsignals[i].shape[1]):
            signal = []
            # loop through the segments (sweeps) and add them to the dict
            for s in range(bl.size['segments']):
                signal.append(np.array(bl.segments[s].analogsignals[i][:, ch]))
            ch_numb += 1
            data_out['Ch{:}'.format(ch_numb)] = signal
    return data_out
