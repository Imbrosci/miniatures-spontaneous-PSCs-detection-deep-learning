# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:21:03 2022

@author: barbara
"""

import numpy as np
import os
import pandas as pd
from scipy.signal import resample
from abf_files_loader import load_abf
from butter import lowpass
from get_data import get_format_data
from display_proof_revise import MiniSpontProof
from txt_files_loader import load_txt
# %

root = os.getcwd() + '/'
results_path = 'results/results.xlsx'
recording_dir = 'recordings/'

check_excel = False
trials = 0

while not check_excel:
    # get file name and channel to proof-read
    recording_file = input('Enter the file name: ')
    channel = input('Enter the channel number: ')

    # define the sheet_name
    if ('.abf' in recording_file) or ('.txt' in recording_file):
        sheet_name = recording_file[:-4] + '_' + channel
    else:
        sheet_name = recording_file + '_' + channel

    # try to load the results
    try:
        results = pd.read_excel(root + results_path, sheet_name=sheet_name)
        check_excel = True
    except ValueError:
        print('''Either the recording file or the channel was not found in the results.xlsx file. Please, try again.''')
        trials += 1
        if trials >= 5:
            exit()

# try to load the summary results sheet
try:
    summary_results = pd.read_excel(root + results_path,
                                    sheet_name='Summary results')
except ValueError:
    print('There is a problem with the results.xlsx file. Make sure there is a summary_results sheet')

# try to load the recording_file
if '.abf' in recording_file:
    try:
        data = load_abf(root + recording_dir + recording_file)
    except FileNotFoundError:
        print('The recording file was not found')
        exit()
if '.txt' in recording_file:
    try:
        data = load_txt(root + recording_dir + recording_file)
    except FileNotFoundError:
        print('The recording file was not found')
        exit()  
        
# get metadata
start_sweep = summary_results.loc[(
    summary_results['Recording filename'] == recording_file) & (
        summary_results['Channel'] == int(channel)
        )]['Analysis start at sweep (number)']

start_at = summary_results.loc[(
    summary_results['Recording filename'] == recording_file) & (
        summary_results['Channel'] == int(channel)
        )]['Starting point at each sweep (ms)']

analysis_end = summary_results.loc[(
    summary_results['Recording filename'] == recording_file) & (
        summary_results['Channel'] == int(channel)
        )]['Analysis length (sec)']

fs = summary_results.loc[(
    summary_results['Recording filename'] == recording_file) & (
        summary_results['Channel'] == int(channel)
        )]['Sampling rate (Hz)']

start_sweep = int(start_sweep - 1)
fs = int(fs)
start_at = int(start_at * (fs / 1000))
analysis_end = int(analysis_end * fs)

# try to get the relevant channel
ch = ['Ch' + str(channel)]
data = get_format_data(data, ch, start_sweep, start_at)
if len(data) == 0:
    print('The channel was not found in the recording file.')
    exit()

# get the portion of the recording been analyzed
signal = data[int(channel)]
signal = signal[:analysis_end]

# resample the signal if fs != 20000
if fs != 20000:
    signal = resample(signal, int(signal.shape[0] * 20000 / fs))

# low pass filter the recording
signal_lp = lowpass(signal, 800, order=1)

# get x, y coordinates of detected events
x = np.array(results['x (ms)'])
y = np.array(results['y (pA)'])

time = np.linspace(0, round(signal_lp.shape[0]/(fs / 1000)) - 1,
                   signal_lp.shape[0])

original_values = np.array(list(zip(x, y)), dtype='object')

# start the proof-reading
mini_spont_proof = MiniSpontProof(root, results_path, sheet_name, time,
                                  signal_lp, original_values, fs)

