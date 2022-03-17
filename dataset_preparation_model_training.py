# -*- coding: utf-8 -*-
"""
Created in Feb 2022.

This code creates the training and validation datasets, train a deep learning
model and test its performance on the validation dataset.

@author: imbroscb
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from dataset_generator import dataset_generator
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn import metrics

# %% create the dataset

root = '/alzheimer/barbara/all_data/miniatures_spontaneous/'
annotations_path = 'annotations/notes.xlsx'
recordings_path = 'recordings/'

x_train, y_train, x_test, y_test = dataset_generator(root, recordings_path,
                                                     annotations_path)

# %% check 1 random example from the train dataset

num = random.randint(1, x_train.shape[0])
plt.plot(x_train[num, :])
plt.plot(y_train[num, :])

plt.title('Label: %d' % num)

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
i = Input(shape=(x_train.shape[1:],))
x = Dense(128, activation='relu')(i)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.35)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(300, activation='sigmoid')(x)

model = Model(i, x)

# compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# callbacks
checkpoint_path = '/alzheimer/barbara/all_data/miniatures_spontaneous/trained_models/model.h5'
# checkpoint_path = '/alzheimer/barbara/all_data/miniatures_spontaneous/model_weights/'
# checkpoint_path = checkpoint_path + 'cp-{epoch:03d}.hdf5' in this case add
# save_freq='epoch' to Modelcheckpoint

callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]

# model.save_weights(checkpoint_path.format(epoch=0))

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

# %% metrics of saved models weights

# second last training (in trained_models (copy 1))
# Recall:  0.9753258845437617
# Precision:  0.8261041009463722
# Specificity:  0.7956441149212233
# F1_score:  0.894534585824082

# last training (shorter length limit rise_time
# Recall:  0.9697392923649907
# Precision:  0.8216962524654833
# Specificity:  0.7905468025949953
# F1_score:  0.8896006833226564

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
