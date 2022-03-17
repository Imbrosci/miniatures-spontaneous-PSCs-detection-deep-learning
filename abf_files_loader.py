# -*- coding: utf-8 -*-

from neo import io
import numpy as np


def load_abf(filename):
    """
    Created on Tue Sep 11 10:52:17 2018

    This function extracts the analog signal(s) from abf files.

    @author: imbroscb
    """
    data = io.AxonIO(filename)
    b1 = data.read()[0]
    chan = {}
    for ch in range(b1.segments[0].analogsignals[0].shape[1]):
        signal = []
        for s in range(len(b1.segments)):
            signal.append(np.array(b1.segments[s].analogsignals[0][:, ch]))
            numb = ch + 1
        chan['ch%d' % numb] = signal
    return chan
