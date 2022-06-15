# -*- coding: utf-8 -*-

import numpy as np


def load_txt(filename, datatype='float'):
    """
    Created on Wed Feb 13 14:43:47 2019

    load txt data of type 'float' and organized them in sweeps

    @author: imbroscb
    """
    if datatype == 'float':
        with open(filename) as f:
            temp = [line.split() for line in f]
            # recording = np.zeros((len(temp), len(temp[0])))
            recording = []
            for c in range(len(temp[0])):
                current_sweep = np.zeros((len(temp), 1))
                for r in range(len(temp)):
                    current_sweep[r] = float(temp[r][c])
                recording.append(current_sweep)
    else:
        recording = []
        with open(filename) as f:
            for line in f:
                recording.append(line.strip())
    
    data = {}
    data['Ch1'] = recording
    return data
