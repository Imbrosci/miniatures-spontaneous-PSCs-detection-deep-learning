# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:23:07 2022.

@author: barbara
"""

import xlsxwriter
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks, resample
from abf_files_loader import load_abf
from butter import lowpass
from get_data import get_format_data
from pscs_parameters_calculator import ParametersCalculator
from models_architecture import ModelBase, ModelRefinement
from txt_files_loader import load_txt

# %  load a file.abf

root = os.getcwd() + '/'
results_dir = 'results/'
recording_dir = 'recordings/'
metadata_path = 'metadata/metadata.xlsx'
model_base_path = 'trained_models/model.h5'
model_refinement_path = 'trained_models/model_refinement.h5'


# get the trained models (load the model and the pre-trained weights)
ModelType = ModelBase()
model_base = ModelType.model
model_base.load_weights(root + model_base_path)
ModelType = ModelRefinement()
model_refinement = ModelType.model
model_refinement.load_weights(root + model_refinement_path)

# %
# get the metadata file
metadata = pd.read_excel(root + metadata_path)

# generate an excel file where to store the results
excelfile_name = 'results.xlsx'
book = xlsxwriter.Workbook(root + results_dir + excelfile_name)
main_result_sheet = book.add_worksheet('Summary results')
mrs_current_row = 0
main_result_sheet.write(mrs_current_row, 0, 'Recording filename')
main_result_sheet.write(mrs_current_row, 1, 'Channel')
main_result_sheet.write(mrs_current_row, 2, 'Sampling rate (Hz)')
main_result_sheet.write(mrs_current_row, 3, 'Analysis start at sweep (number)')
main_result_sheet.write(mrs_current_row, 4, 'Cut sweeps first part (ms)')
main_result_sheet.write(mrs_current_row, 5, 'Cut sweeps last part (ms)')
main_result_sheet.write(mrs_current_row, 6, 'Analysis length (sec)')
main_result_sheet.write(mrs_current_row, 7, 'Average interevent interval (ms)')
main_result_sheet.write(mrs_current_row, 8, 'Average amplitude (pA)')
main_result_sheet.write(mrs_current_row, 9, 'Average 10-90% rise time (ms)')
main_result_sheet.write(mrs_current_row, 10, 'Average 90-10% decay time (ms)')
main_result_sheet.write(mrs_current_row, 11, 'Stdev of the baseline signal (pA)')
main_result_sheet.write(mrs_current_row, 12, 'Manually revised')

for result_file in os.listdir(root + results_dir):
    if result_file == excelfile_name:
        print('')
        print('''There is already an excel file with the same name. Do you want to overwrite it? (yes/no)''')
        check_answer = False
        while not check_answer:
            answer = input()
            if answer == 'yes':
                check_answer = True
            elif answer == 'no':
                print('''Please, move the results.xlsx file to another directory and start the analysis again.''')
                print('')
                check_answer = True
                exit()
            else:
                print('''The answer should be yes or no. Please enter your answer again.''')

# get the number of recordings to analyse
analysis_numb = 0
for index, row in metadata.iterrows():
    channels = []
    if type(row['Channels to use']) == int:
        channels.append(row['Channels to use'])
    elif type(row['Channels to use']) == float:
        channels.append(int(row['Channels to use']))
    else:
        for c in row['Channels to use'].split(','):
            channels.append(int(c))
    analysis_numb = analysis_numb + len(channels)

counter = 1

# loop through the rows of metadata.xslx
for index, row in metadata.iterrows():
    # get information from the metadata file
    recording_file = row['Name of recording']
    channels = []
    if type(row['Channels to use']) == int:
        channels.append(row['Channels to use'])
    else:
        for c in row['Channels to use'].split(','):
            channels.append(int(c))

    # load the recordings
    if recording_file[-4:] == '.abf':
        data = load_abf(root + recording_dir + recording_file)
        ch = []
        for k in data.keys():
            if int(k[2:]) in channels:
                ch.append(k)

    if recording_file[-4:] == '.txt':
        data = load_txt(root + recording_dir + recording_file)
        ch = 'Ch1'

    # get information from a row of the metadata.xlsx file
    fs = row['Sampling rate (Hz)']
    analysis_end = int(row['Analysis length (min)'] * 60 * fs)

    for k, v in data.items():
        sweep_length = v[0].shape[0]
    
    first_sweep = int(row['Analysis start at sweep (number)']) - 1
    first_point = int(row['Cut sweeps first part (ms)'] * (fs / 1000))
    last_point = int(sweep_length - row['Cut sweeps last part (ms)'] * (
        fs / 1000))
    
    # get the recordings channels to be analyzed with concatenated sweeps
    # the sweeps may be cutted according to first_point and last_point
    data = get_format_data(data, ch, first_sweep, first_point, last_point)

    print('-------------------------------------------------')
    print('The analysis of the file {} is starting'.format(recording_file))
    print('')
    print('The following details are used:')
    for col in range(len(row)):
        if col < 2:
            print(row.index[col] + ':', row[col])
        elif col == 2:
            print(row.index[col][:-5] + ':',
                  int(row[col]), row.index[col][-3:-1])
        elif col == 3:
            print(row.index[col][:-9] + ' number: ', int(row[col]))
        elif(col == 4) or (col == 5):
            print(row.index[col][:-5] + ':', int(row[col]),
                  row.index[col][-3:-1])
        else:
            print(row.index[col][:-6] + ':', int(row[col]),
                  row.index[col][-4:-1])

    print('')
    if len(data) == 0:
        print('The selected channel(s) {:} was/were not found in the recording file.'.format(tuple(channels)))
        continue

    # analysis of each recording channel
    for ch, signal in data.items():

        print('Running analysis of channel {} ({}/{})'.format(ch, counter,
                                                              analysis_numb))
        counter += 1
        mrs_current_row += 1
        main_result_sheet.write(mrs_current_row, 0, recording_file)
        main_result_sheet.write(mrs_current_row, 1, ch)
        main_result_sheet.write(mrs_current_row, 2, fs)
        main_result_sheet.write(mrs_current_row, 3, first_sweep + 1)
        main_result_sheet.write(mrs_current_row, 4,
                                int(first_point / fs * 1000))
        main_result_sheet.write(mrs_current_row, 5,
                                int((sweep_length - last_point) / fs * 1000))

        # get the portion of recording to analyse
        analysis_end = min(analysis_end, len(signal))
        signal = signal[:analysis_end]

        # resample the signal if fs != 20000
        if fs != 20000:
            signal = resample(signal, int(signal.shape[0] * 20000 / fs))

        # low pass filter the signal
        signal_lp = lowpass(signal, 800, order=1)

        # z-score the signal
        signal_z_scored = np.zeros((signal_lp.shape))
        signal_z_scored = (signal_lp - np.mean(signal_lp[10:, 0])) / np.std(
            signal_lp[10:, 0])

        # straighten the signal (to improve the model performance)
        signal_length = int(signal_z_scored.shape[0])
        chunk_length = int(30 * fs)
        for s in range(int(signal_length // chunk_length + 1)):
            chunk_start = int(s * chunk_length)
            chunk_end = int(chunk_start + chunk_length)

            if chunk_end >= signal_length:
                chunk_end = signal_length
                chunk_length = chunk_end - chunk_start

            if chunk_length == 0:
                continue

            shift = (
                np.mean(
                    signal_z_scored[chunk_start + 10:
                                    chunk_start + 10000]) - np.mean(
                                        signal_z_scored[chunk_end - 10000:
                                                        chunk_end])
                                            ) / chunk_length
            shift_pd = pd.Series(np.linspace(shift, shift, chunk_length))
            cum_shift = np.array(shift_pd.cumsum()).reshape(-1, 1)
            signal_z_scored[chunk_start:chunk_end] = np.sum(
                [signal_z_scored[chunk_start:chunk_end], cum_shift], axis=0)
            signal_z_scored[chunk_start:chunk_end] = signal_z_scored[
                chunk_start:chunk_end] - np.mean(
                    signal_z_scored[chunk_start:chunk_end], axis=0)

        # generate the prediction_trace
        border_left = 0
        input_data = np.zeros((int(signal_z_scored.shape[0] / 300), 300))
        for win in range(int(signal_z_scored.shape[0]/300) + 1):
            border_right = border_left + 300
            if border_right > signal_z_scored.shape[0]:
                break
            data_win_300 = signal_z_scored[border_left:border_right, 0]
            data_win_300 = data_win_300.astype('float64')
            input_data[win, :] = data_win_300
            border_left = border_left + 300

        pred = model_base.predict(input_data)
        prediction_trace = pred.reshape(-1,)
        prediction_trace = lowpass(prediction_trace, 500)

        # get the events using two threshold on the prediction trace and some
        # additional steps
        thr1 = 0.20
        thr2 = 0.10
        t = 0
        pred_peaks = []
        x_pre = []
        x_dp_pre = []
        y_pre = []
        numb = 0
        for p in range(prediction_trace.shape[0]):
            current_pred = prediction_trace[p, ]
            current_pA = signal_lp[p, ]
            if (current_pred >= thr1) and (p >= t):
                iceberg_pred = []
                iceberg_pA = []
                t = p + 1
                while (current_pred > thr2) and (
                        t < prediction_trace.shape[0]):
                    iceberg_pred.append(current_pred)
                    iceberg_pA.append(current_pA)
                    current_pred = prediction_trace[t]
                    current_pA = signal_lp[t]
                    t += 1
                numb += 1

                # look for peaks in each iceberg found in the prediction trace
                peak_y = 0
                biggest = None
                for peak_x in find_peaks(iceberg_pred)[0]:
                    if iceberg_pred[peak_x] > peak_y:
                        peak_y = iceberg_pred[peak_x]
                        biggest_peak = peak_x

                two_ms_dp = int(fs / 500)
                one_ms_dp = int(fs / 1000)

                for peak_x in find_peaks(iceberg_pred)[0]:

                    # the biggest peak is appended
                    if peak_x == biggest_peak:
                        uncorrect_peak_x = peak_x + p
                        delta_left = min(one_ms_dp, uncorrect_peak_x)
                        delta_right = min(
                            one_ms_dp, signal_lp.shape[0] - uncorrect_peak_x)
                        correction = np.argmin(
                            signal_lp[uncorrect_peak_x - delta_left:
                                      uncorrect_peak_x + delta_right]
                                ) - delta_left
                        correct_peak_x = uncorrect_peak_x + correction
                        pred_peaks.append(prediction_trace[correct_peak_x])
                        x_pre.append(correct_peak_x / one_ms_dp)
                        x_dp_pre.append(correct_peak_x)
                        y_pre.append(signal_lp[correct_peak_x])

                    # the smaller peaks are appended if the conditions
                    # below are met
                    else:
                        delta_left = min(two_ms_dp, peak_x)
                        delta_right = min(two_ms_dp,
                                          len(iceberg_pred) - peak_x)
                        eighty_perc = iceberg_pred[peak_x] * 0.8

                        # condition to append a peak
                        if (np.min(
                                iceberg_pred[peak_x - delta_left:peak_x]
                                ) < eighty_perc) and (np.min(
                                    iceberg_pred[peak_x:peak_x + delta_right]
                                    ) < eighty_perc):
                            uncorrect_peak_x = peak_x + p
                            delta_left = min(one_ms_dp, uncorrect_peak_x)
                            delta_right = min(
                                one_ms_dp,
                                signal_lp.shape[0] - uncorrect_peak_x)
                            correction = np.argmin(
                                signal_lp[uncorrect_peak_x - delta_left:
                                          uncorrect_peak_x + delta_right]
                                    ) - delta_left
                            correct_peak_x = uncorrect_peak_x + correction
                            pred_peaks.append(prediction_trace[correct_peak_x])
                            x_pre.append(correct_peak_x / one_ms_dp)
                            x_dp_pre.append(correct_peak_x)
                            y_pre.append(signal_lp[correct_peak_x])

        # exclude presumable false positives with predictions from
        # model_refinement
        x = []
        x_dp = []
        y = []
        for i in range(len(x_dp_pre)):
            delta_left = min(150, x_dp_pre[i])
            delta_right = min(150, signal_z_scored.shape[0] - x_dp_pre[i])
            data_win_300 = signal_z_scored[x_dp_pre[i] - delta_left:
                                           x_dp_pre[i] + delta_right, 0]
            data_win_300 = data_win_300.astype('float64')
            if data_win_300.shape[0] == 300:
                pred = model_refinement.predict(data_win_300.reshape(1, -1))
                if pred >= 0.25:
                    x.append(x_pre[i])
                    x_dp.append(x_dp_pre[i])
                    y.append(y_pre[i])

        # calculate the interevent interval
        x_pd = pd.Series(x)
        interevent_interval = x_pd.shift(-1) - x_pd

        # instantiate parameters
        parameters = ParametersCalculator(signal_lp, x_dp, y, fs)

        # calculate the amplitude
        amplitude = parameters.amplitude_stdev()[0]

        # save the results to excel file
        if ('.abf' in recording_file) or ('.txt' in recording_file):
            sheet_name = recording_file[:-4] + '_' + str(ch)
        else:
            sheet_name = recording_file + '_' + str(ch)

        current_sheet = book.add_worksheet(sheet_name)
        current_sheet.write(0, 0, 'x (ms)')
        current_sheet.write(0, 1, 'y (pA)')
        current_sheet.write(0, 2, 'Interevent interval (ms)')
        current_sheet.write(0, 3, 'Amplitude (pA)')

        for ev_idx in range(len(x)):
            current_sheet.write(ev_idx + 1, 0, x[ev_idx])
            current_sheet.write(ev_idx + 1, 1, y[ev_idx])
            try:
                current_sheet.write(ev_idx + 1, 2, interevent_interval[ev_idx])
            except TypeError:
                pass
            try:
                current_sheet.write(ev_idx + 1, 3, amplitude[ev_idx])
            except TypeError:
                pass

        # get the average interevent interval and amplitude
        average_interevent_interval = np.nanmean(interevent_interval)
        average_amplitude = np.nanmean(amplitude)

        # write the average interevent interval and amplitude
        main_result_sheet.write(mrs_current_row, 6, signal_lp.shape[0] / fs)
        main_result_sheet.write(mrs_current_row, 7,
                                average_interevent_interval)
        main_result_sheet.write(mrs_current_row, 8, average_amplitude)

        # calculate the mean rise (10-90%) and the mean decay time (90-10%)
        mean_rise_time = parameters.rise_time()[0]
        mean_decay_time = parameters.decay_time()[0]

        # calculate the stdev
        stdev = parameters.amplitude_stdev()[1]
        average_stdev = np.nanmean(stdev)

        # write the mean rise and decay time and stdev
        try:
            main_result_sheet.write(mrs_current_row, 9, mean_rise_time)
        except TypeError:
            pass
        try:
            main_result_sheet.write(mrs_current_row, 10, mean_decay_time)
        except TypeError:
            pass
        try:
            main_result_sheet.write(mrs_current_row, 11, average_stdev)
        except TypeError:
            pass

        # write that the results are not revised yet
        main_result_sheet.write(mrs_current_row, 12, 'no')

book.close()
