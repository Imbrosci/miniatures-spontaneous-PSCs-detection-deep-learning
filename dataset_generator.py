# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:47:31 2022.

@author: imbroscb
"""

import numpy as np
import pandas as pd
import copy
from abf_files_loader import load_abf
from get_channels import get_channels
from butter import lowpass


def dataset_generator_base(root, recordings_path, annotations_path):
    """
    Create the training (85%) and validation (15%) datasets.

    The datasets should be used for training a model to detect miniature and
    spontaneous postsnyptic currents (mPSCs and sPSCs, respectively).

    The datasets represent short sequences (15 ms long) of currents (pA) from
    patch-clamp recorded neurons.

    The labels are binary vectors of the same length, filled with ones around
    the trough of an identified event and with zeros otherwise.
    Events are manually annotated. The manual annotations are limited to one
    time point per event (ideally close to the trough) but this function
    changes this by filling the labels with ones at more time points per event.

    The function assumes a sampling rate of 20 kHz.

    @author: barbara
    """
    annotations_summary = pd.read_excel(root + annotations_path,
                                        sheet_name='Summary')
    first_round = 1

    for r, rec in annotations_summary.iterrows():

        file = rec[0]
        channel = rec[1]
        sheetname = rec[2]
        try:
            sweep_start = int(rec[4])
        except ValueError:
            pass

        model_to_train = rec[3]

        if model_to_train != 'model':
            continue

        print('writing annotations for {}'.format(sheetname))
        annotations = pd.read_excel(root + annotations_path,
                                    sheet_name=sheetname)
        # get annotations
        x = np.array(annotations.iloc[:, 0])
        y = np.array(annotations.iloc[:, 1])

        selected_trace = []
        if annotations.shape[1] > 2:
            for row in range(annotations.shape[0]):
                if not annotations.iloc[row, :].isnull()[3]:
                    selected_trace.append([int(annotations.iloc[row, 2]),
                                           int(annotations.iloc[row, 3])])

        # create a tuple with x and y
        xy_coord = (x, y)

        # get a recording
        data = load_abf(root + recordings_path + file)

        # get the channels
        keys = []
        channels = []
        for k, v in data.items():
            keys.append(k)
            channels.append(int(k[-1]))

        # get data organized and selected
        data = get_channels(data, keys, channels, start_at=sweep_start)

        if len(selected_trace) > 0:
            selected_data = {}
            for ch, values in data.items():
                counter = 0
                for s in selected_trace:
                    if counter == 0:
                        selected_data_ch = values[s[0]:s[1]]
                    else:
                        temp = values[s[0]:s[1]]
                        selected_data_ch = np.concatenate((selected_data_ch,
                                                           temp))
                    counter += 1
                selected_data[ch] = selected_data_ch

            data = selected_data

        # filter out the very high frequency component from one channel
        data_lp = lowpass(data[channel], 800, order=1).reshape(-1, )

        # get only the part of recordings with annotation
        last_dp = x[-1] + 1000
        data_lp = data_lp[:last_dp, ].reshape(-1, 1)

        # z-score signal
        data_z_scored = np.zeros((data_lp.shape))
        for sw in range(data_lp.shape[1]):
            data_z_scored[:, sw] = (
                data_lp[:, sw] - np.mean(
                    data_lp[10000:, sw])) / np.std(data_lp[10000:, sw])

        # generation of chunks for training dataset
        pos_traces = []
        neg_traces = []
        binary_pos_traces = []
        binary_neg_traces = []

        for sw in range(1):  # change if the data are organized in more sweeps

            min_range = 0
            max_range = min_range + 300
            for win in range(int(data_z_scored.shape[0] / 60)):

                timestamp_check = range(min_range, max_range)
                binary = np.zeros((300,))
                found = 0
                for t in range(300):
                    if timestamp_check[t] in xy_coord[0][:]:
                        found += 1
                        # t_p = timestamp_check[t]
                        peak = np.argmin(data_z_scored[
                            min_range + t - 15:min_range + t + 25, sw])
                        peak = t - 15 + peak
                        if peak < 0:
                            peak = 0
                        if peak > 300:
                            peak = 300
                        abs_peak = min_range + peak
                        if abs_peak < 10000:
                            av = np.mean(data_z_scored[:abs_peak + 10000])
                            st_dev = np.std(data_z_scored[:abs_peak + 10000])
                        elif abs_peak + 10000 >= data_z_scored.shape[0]:
                            av = np.mean(data_z_scored[abs_peak - 10000:])
                            st_dev = np.std(data_z_scored[abs_peak - 10000:])
                        else:
                            av = np.mean(data_z_scored[abs_peak - 10000:
                                                       abs_peak + 10000])
                            st_dev = np.std(data_z_scored[abs_peak - 10000:
                                                          abs_peak + 10000])
                        thr = av - st_dev

                        current_y = data_z_scored[abs_peak]
                        dp = -1
                        start = peak
                        while current_y < thr:
                            current_y = data_z_scored[abs_peak + dp]
                            start = peak + dp
                            dp -= 1
                            if (abs_peak - dp < 0):
                                break

                        current_y = data_z_scored[abs_peak]
                        dp = 1
                        end = peak
                        while current_y < thr:
                            current_y = data_z_scored[abs_peak + dp]
                            end = peak + dp
                            dp += 1
                            if (abs_peak + dp >= data_z_scored.shape[0]):
                                break

                        # define rise and decay time
                        rise_time = peak - start
                        decay_time = end - peak

                        # limit rise and decay time length
                        if rise_time > 40:  # 60
                            rise_time = 40
                        elif rise_time < 15:
                            rise_time = 15
                        if decay_time > 70:  # 100
                            decay_time = 70
                        elif decay_time < 15:
                            decay_time = 15

                        rise_label = np.linspace(1, 1, rise_time)
                        decay_label = np.linspace(1, 1, decay_time)

                        if start >= 0:
                            rise_decay_label = np.concatenate((rise_label,
                                                               decay_label))

                            if len(rise_decay_label) >= 300 - start:
                                rise_decay_label = rise_decay_label[
                                    :300 - start]

                            binary[start:start + len(rise_decay_label)
                                   ] = rise_decay_label

                        else:
                            rise_label = rise_label[np.abs(start):]
                            rise_decay_label = np.concatenate((rise_label,
                                                               decay_label))

                            if len(rise_decay_label) >= 300:
                                rise_decay_label = rise_decay_label[:300]

                            binary[:len(rise_decay_label)] = rise_decay_label

                if found > 0:
                    if data_z_scored[min_range:max_range, sw].shape[0] == 300:
                        pos_traces.append(data_z_scored[min_range:max_range,
                                                        sw])
                        binary_pos_traces.append(binary)

                else:
                    if data_z_scored[min_range:max_range, sw].shape[0] == 300:
                        neg_traces.append(data_z_scored[
                            min_range:max_range, sw])
                        binary_neg_traces.append(binary)

                min_range = min_range + 60
                max_range = min_range + 300

        np_pos_traces = np.array(pos_traces)
        np_neg_traces = np.array(neg_traces)[:np_pos_traces.shape[0]]
        np_pos_binary_traces = np.array(binary_pos_traces)
        np_neg_binary_traces = np.array(binary_neg_traces)[
            :np_pos_traces.shape[0]]

        N = np_pos_traces.shape[0]
        np_pos_traces_train = np_pos_traces[:int(N / 10 * 8.5)]
        np_pos_traces_test = np_pos_traces[int(N / 10 * 8.5):]
        np_neg_traces_train = np_neg_traces[:int(N / 10 * 8.5)]
        np_neg_traces_test = np_neg_traces[int(N / 10 * 8.5):]

        np_pos_binary_traces_train = np_pos_binary_traces[:int(N / 10 * 8.5)]
        np_pos_binary_traces_test = np_pos_binary_traces[int(N / 10 * 8.5):]
        np_neg_binary_traces_train = np_neg_binary_traces[:int(N / 10 * 8.5)]
        np_neg_binary_traces_test = np_neg_binary_traces[int(N / 10 * 8.5):]

        x_train_pre = np.concatenate((np_pos_traces_train,
                                      np_neg_traces_train))
        x_test_pre = np.concatenate((np_pos_traces_test, np_neg_traces_test))

        y_train_pre = np.concatenate((np_pos_binary_traces_train,
                                      np_neg_binary_traces_train))
        y_test_pre = np.concatenate((np_pos_binary_traces_test,
                                     np_neg_binary_traces_test))

        if first_round == 1:
            x_train = copy.deepcopy(x_train_pre)
            x_test = copy.deepcopy(x_test_pre)
            y_train = copy.deepcopy(y_train_pre)
            y_test = copy.deepcopy(y_test_pre)
            first_round = 0

        else:
            x_train = np.concatenate((x_train, x_train_pre))
            x_test = np.concatenate((x_test, x_test_pre))
            y_train = np.concatenate((y_train, y_train_pre))
            y_test = np.concatenate((y_test, y_test_pre))

    return x_train, y_train, x_test, y_test
