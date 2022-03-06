# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:47:31 2022

- create the train and test (validation) dataset
# train the model and perform first model evaluation

@author: imbroscb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from butter import lowpass
from abf_files_loader import load_abf
from get_channels import get_channels
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn import metrics

# %%

root = '/alzheimer/barbara/all_data/miniatures_spontaneous/'
annotations_path = 'annotations/notes.xlsx'
recordings_path = 'recordings/'
annotations_summary = pd.read_excel(root + annotations_path,
                                    sheet_name='Summary')

first_round = 1

for r, rec in annotations_summary.iterrows():

    file = rec[0]
    channel = rec[1]
    sheetname = rec[2]
    try:
        sweep_start = int(rec[5])
    except:
        pass

    model_to_train = rec[3]

    if model_to_train == 'model_refinement':
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
            if annotations.iloc[row, :].isnull()[3] == False:
                selected_trace.append([int(annotations.iloc[row, 2]),
                                        int(annotations.iloc[row, 3])])

    # create a tuple with x and y
    xy_coord = (x, y)

    # get recording
    data = load_abf(root + recordings_path + file)

    #get channels
    channels = []
    for k, v in data.items():
        channels.append(int(k[-1]))

    # get data organized and selected
    data = get_channels(data, channels, fs=20000, start=sweep_start)

    if len(selected_trace) > 0:
        selected_data = {}
        for ch, values in data.items():
            counter = 0
            for s in selected_trace:
                if counter == 0:
                    selected_data_ch = values[s[0]:s[1]]
                else:
                    temp = values[s[0]:s[1]]
                    selected_data_ch = np.concatenate((selected_data_ch, temp))
                counter += 1
            selected_data[ch] = selected_data_ch

        data = selected_data

    # filter out the very high frequency component from one channel
    data_lp = lowpass(data[channel], 800, order=1).reshape(-1, )

    # get only the part of recordings with annotation
    last_dp = x[-1] + 1000
    data_lp = data_lp[:last_dp,].reshape(-1, 1)

    # z-score signal
    data_z_scored = np.zeros((data_lp.shape))
    for sw in range(data_lp.shape[1]):
        data_z_scored[:, sw] = (data_lp[:, sw] - np.mean(data_lp[10000:, sw])
                                ) / np.std(data_lp[10000:, sw])

    # generation of chunks for training dataset
    pos_traces = []
    neg_traces = []
    binary_pos_traces = []
    binary_neg_traces = []

    for sw in range(1):  # change if the data are organized in more sweeps
        print(sw)
        min_range = 0
        max_range = min_range + 300
        for win in range(int(data_z_scored.shape[0] / 60)):

            timestamp_check = range(min_range, max_range)
            binary = np.zeros((300,))
            found = 0
            for t in range(300):
                if timestamp_check[t] in xy_coord[0][:]:
                    found += 1
                    t_p = timestamp_check[t]
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
                        av= np.mean(
                            data_z_scored[abs_peak - 10000:abs_peak + 10000])
                        st_dev= np.std(
                            data_z_scored[abs_peak - 10000:abs_peak + 10000])

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

                    rise_time = peak - start
                    decay_time = end - peak
                    if rise_time > 60:
                        rise_time = 60
                    elif rise_time < 15:
                        rise_time = 15
                    if decay_time > 100:
                        decay_time = 100
                    elif decay_time < 15:
                        decay_time = 15

                    rise_label = np.linspace(1, 1, rise_time)
                    decay_label = np.linspace(1, 1, decay_time)

                    if start >= 0:
                        rise_decay_label = np.concatenate((rise_label,
                                                           decay_label))
                    # if end > 300:
                        if len(rise_decay_label) >= 300 - start:
                            rise_decay_label = rise_decay_label[:300 - start]

                        binary[start:start + len(rise_decay_label)
                               ] = rise_decay_label

                    else:
                        rise_label = rise_label[np.abs(start):]
                        rise_decay_label = np.concatenate((rise_label,
                                                           decay_label))
                        # if end > 300:
                        if len(rise_decay_label) >= 300:
                            rise_decay_label = rise_decay_label[:300]

                        binary[:len(rise_decay_label)] = rise_decay_label

            if found > 0:
                if found > 1:
                    print(len(pos_traces))
                if data_z_scored[min_range:max_range, sw].shape[0] == 300:
                    pos_traces.append(data_z_scored[min_range:max_range, sw])
                    binary_pos_traces.append(binary)
                # if cc == 9:  # rm
                #    print(t_p) # rm
                # cc += 1 # rm
            else:
                if data_z_scored[min_range:max_range, sw].shape[0] == 300:
                    neg_traces.append(data_z_scored[min_range:max_range, sw])
                    binary_neg_traces.append(binary)

            min_range = min_range + 60
            max_range = min_range + 300

    np_pos_traces = np.array(pos_traces)
    np_neg_traces = np.array(neg_traces)[:np_pos_traces.shape[0]]
    np_pos_binary_traces = np.array(binary_pos_traces)
    np_neg_binary_traces = np.array(binary_neg_traces)[:np_pos_traces.shape[0]]
    pos_labels = np.linspace(1, 1, np_pos_traces.shape[0])
    neg_labels = np.linspace(0, 0, np_neg_traces.shape[0])

    N = np_pos_traces.shape[0]
    np_pos_traces_train = np_pos_traces[:int(N / 10 * 8)]
    np_pos_traces_test = np_pos_traces[int(N / 10 * 8):]
    np_neg_traces_train = np_neg_traces[:int(N / 10 * 8)]
    np_neg_traces_test = np_neg_traces[int(N / 10 * 8):]

    np_pos_binary_traces_train = np_pos_binary_traces[:int(N / 10 * 8)]
    np_pos_binary_traces_test = np_pos_binary_traces[int(N / 10 * 8):]
    np_neg_binary_traces_train = np_neg_binary_traces[:int(N / 10 * 8)]
    np_neg_binary_traces_test = np_neg_binary_traces[int(N / 10 * 8):]

    pos_labels_train = pos_labels[:int(N / 10 * 8)]
    pos_labels_test = pos_labels[int(N / 10 * 8):]
    neg_labels_train = neg_labels[:int(N / 10 * 8)]
    neg_labels_test = neg_labels[int(N / 10 * 8):]

    x_train_pre = np.concatenate((np_pos_traces_train, np_neg_traces_train))
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

# %% check 1 random example from the train dataset

num = random.randint(1, x_train.shape[0])
plt.plot(x_train[num, :])
plt.plot(y_train[num, :])

plt.title('Label: %d' %num)

# %% shuffle the train and test dataset

# train dataset
samples_list = list(range(x_train.shape[0]))
random.shuffle(samples_list)
x_shuffled_train = np.zeros((x_train.shape))
y_shuffled_train = np.zeros((y_train.shape))

for smp in range(x_train.shape[0]):
    x_shuffled_train[smp, :] = x_train[samples_list[smp], :]
    y_shuffled_train[smp, :] = y_train[samples_list[smp], :]

# test dataset
samples_list = list(range(x_test.shape[0]))
random.shuffle(samples_list)
x_shuffled_test = np.zeros((x_test.shape))
y_shuffled_test = np.zeros((y_test.shape))

for smp in range(x_test.shape[0]):
    x_shuffled_test[smp, :] = x_test[samples_list[smp], :]
    y_shuffled_test[smp, :] = y_test[samples_list[smp], :]

x_train = x_shuffled_train
x_test = x_shuffled_test
y_train = y_shuffled_train
y_test = y_shuffled_test

# %% check 25 random examples from the train dataset

for i in range(25):
    numb = np.random.randint(x_train.shape[0])
    plt.subplot(5, 5, i + 1)
    plt.plot(x_train[numb, :])
    plt.plot(y_train[numb, :])

# %% build and compile the model and get the callbacks

# build the model
i = Input(shape=(x_train.shape[1:]))
x = Dense(32, activation='relu')(i)
x = Dropout(0.1)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(i)
x = Dropout(0.2)(x)
x = Dense(300, activation='sigmoid')(x)

model = Model(i, x)

# compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# % callbacks
checkpoint_path = 'model_weights.h5'
callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                verbose=1, save_best_only=True,
                                                save_weights_only=True,
                                                save_freq='epoch'),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]



# %% fit the model and plot the loss after the training

r = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=512, callbacks=callbacks, epochs=1000)

# plot the loss
plt.figure()
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# %% set the threshold probability in the prediciton for detecting one event
# and load the best model (weights)

thr = 0.2
model.load_weights(checkpoint_path)

# %% display some examples

plt.figure(figsize=(14, 8))
for i in range(10):
    numb = np.random.randint(x_test.shape[0])
    pred = model.predict(x_test[numb, :].reshape(1, -1))
    if np.max(pred) >= 0.2:
        label = 1
    else:
        label = 0
    plt.subplot(5, 2, i+1)
    plt.plot(x_test[numb, :], c='black')
    plt.ylim([-3, 3])
    plt.plot(pred.T, c='orange')
    plt.plot(y_test[numb, :], c='grey')
    plt.axis('off')
    if (np.max(pred) >= thr) and np.sum(y_test[numb, :] > 0):
        plt.title('max. pr.: {:.2f}'.format(np.max(pred)), c='green')
    elif (np.max(pred) < thr) and np.sum(y_test[numb, :] == 0):
        plt.title('max. pr.: {:.2f}'.format(np.max(pred)), c='green')
    else:
        plt.title('max. pr.: {:.2f}'.format(np.max(pred)), c='red')
        
    plt.xticks([])
    plt.yticks([])

# %% calculate metrics

x_test_pos = np.zeros((x_test.shape))
y_test_pos = np.zeros((y_test.shape))

j = 0
for i in range(y_test.shape[0]):
    if np.sum(y_test[i, :]) > 0:
        x_test_pos[j, :] = x_test[i, :]
        y_test_pos[j, :] = y_test[i, :]
        j += 1

x_test_pos = x_test_pos[:j, :]
y_test_pos = y_test_pos[:j, :]

pred = model.predict(x_test_pos)
pred = pred.round()
accuracy_pos = metrics.accuracy_score(y_test_pos, pred)
# print('accuracy_pos: ', accuracy_pos)

pred = model.predict(x_test_pos)

tps = 0
fns = 0 
for p in pred:
    if np.max(p) > thr:
        tps += 1
    else:
        fns += 1

recall = tps / (tps + fns)
print('Recall: ', recall)

x_test_neg = np.zeros((x_test.shape))
y_test_neg = np.zeros((y_test.shape))

j = 0
for i in range(y_test.shape[0]):
    if np.sum(y_test[i, :]) == 0.0:
        x_test_neg[j, :] = x_test[i, :]
        y_test_neg[j, :] = y_test[i, :]
        j += 1

x_test_neg = x_test_neg[:j, :]
y_test_neg = y_test_neg[:j, :]

pred = model.predict(x_test_neg)
pred = pred.round()
accuracy_neg = metrics.accuracy_score(y_test_neg, pred)
# print('accuracy_neg: ', accuracy_neg)

pred = model.predict(x_test_neg)

tns = 0
fps = 0 
for p in pred:
    if np.max(p) > thr:
        fps += 1
    else:
        tns += 1

precision = tps / (tps + fps)
print('Precision: ', precision)

specificity = tns / (tns + fps)
print('Specificity: ', specificity)

f1_score = (2 * precision * recall) / (precision + recall)
print('F1_score: ', f1_score)
