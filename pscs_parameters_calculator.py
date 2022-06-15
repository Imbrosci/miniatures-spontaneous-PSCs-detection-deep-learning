# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:22:29 2022.

@author: barbara
"""

import copy
import numpy as np


class ParametersCalculator():
    """Calculate PSCs amplitude, rise time and decay time."""

    def __init__(self, signal_lp, x_dp, y, fs):
        self.signal_lp = signal_lp
        self.x_dp = x_dp
        self.y = y
        self.fs = fs
        self.amplitude_stdev()
        self.rise_time()
        self.decay_time()

    def amplitude_stdev(self):
        """
        Calculate the amplitude of all PSCs and the stdev of the baseline sig.

        Each event trough is subtracted to a local baseline.
        """
        trace_within_events_dp = []
        for ev in self.x_dp:
            delta_left = int(min((self.fs / 1000) * 10, ev))
            delta_right = int(
                min((self.fs / 1000) * 50, self.signal_lp.shape[0] - (ev + 1)))
            delta_left = max(0, delta_left)
            delta_right = max(0, delta_right)
            temp = np.linspace(ev - delta_left, ev + delta_right - 1,
                               delta_left + delta_right).astype('int')
            trace_within_events_dp.extend(temp)

        signal_wo_events = copy.deepcopy(self.signal_lp)
        signal_wo_events[trace_within_events_dp] = np.nan
        signal_wo_events = signal_wo_events.astype('float64')

        # get the amplitude of the detected events
        amplitude = np.zeros((len(self.x_dp)))
        for ev_idx in range(len(self.x_dp)):
            delta_left = int(min(self.fs, self.x_dp[ev_idx]))
            delta_right = int(
                min(self.fs, signal_wo_events.shape[0] - self.x_dp[ev_idx]))
            local_baseline = np.nanmean(signal_wo_events[
                self.x_dp[ev_idx] - delta_left:
                    self.x_dp[ev_idx] + delta_right])
            amplitude[ev_idx] = local_baseline - self.y[ev_idx]

        stdev = []
        # get a stdev value from 100 ms local signal every sec
        for seg in range(signal_wo_events.shape[0] // self.fs):
            delta_left = int(min((self.fs // 1000) * 50, seg * self.fs))
            delta_right = int(min((self.fs / 1000) * 50,
                                  signal_wo_events.shape[0] - seg * self.fs))
            current_stdev = np.nanstd(
                signal_wo_events[seg * self.fs - delta_left:
                                 seg * self.fs + delta_right])
            stdev.append(current_stdev)

        return amplitude, stdev

    def rise_time(self):
        """
        Calculate the 10-90% rise time.

        To calculate the rise time the average of the detected PSCs are used.
        Cutouts of PSCs (20 ms up to the trough) are used to calculate the
        average signal. PSCs close to the beginning of the recording (<20ms)
        are excluded from calculating the average signal.
        """
        cutouts_rise = []
        x_dp = self.x_dp
        for i in range(len(x_dp)):

            # create cutouts from the rise phase
            steps = int(self.fs / 1000 * 20)
            delta_left = int(min(steps, x_dp[i]))
            ydata = self.signal_lp[x_dp[i] - delta_left:x_dp[i]].reshape(-1,)

            if len(ydata) == steps:
                cutouts_rise.append(ydata)

        cutout_rise_mean = np.mean(np.array(cutouts_rise), axis=0)

        # find the values range
        crm_range = np.max(cutout_rise_mean) - np.min(cutout_rise_mean)

        # normalize the mean trace between 0 and 1 and invert it
        crm_scaled = (cutout_rise_mean - np.min(cutout_rise_mean)) / crm_range
        crm_scaled_inv = 1 - crm_scaled

        # get datapoints number (dp) within 10-90 %
        found_10 = False
        dp_10 = None
        dp_90 = None
        dp_rev_idx = len(crm_scaled_inv) - 1
        for _ in range(len(crm_scaled_inv) - 1):
            if crm_scaled_inv[dp_rev_idx] >= 0.9:
                if crm_scaled_inv[dp_rev_idx - 1] < 0.9:
                    dp_90 = dp_rev_idx
            if crm_scaled_inv[dp_rev_idx] >= 0.1:
                if (crm_scaled_inv[dp_rev_idx - 1] < 0.1) and (not found_10):
                    dp_10 = dp_rev_idx
                    found_10 = True
            dp_rev_idx -= 1

        try:
            rise_time_dp = dp_90 - dp_10

            # calculate the rise time (10-90 %)
            rise_time = rise_time_dp / (self.fs / 1000)

        except TypeError:
            rise_time = np.nan

        return rise_time, dp_10, dp_90

    def decay_time(self):
        """
        Calculate the 90-10% decay time.

        To calculate the decay time the average of the detected PSCs are used.
        Cutouts of PSCs (50 ms starting from the trough) are used to calculate
        the average signal. PSCs close to the end of the recording (<50ms) are
        excluded from calculating the average signal.
        """
        cutouts_decay = []
        x_dp = self.x_dp
        for i in range(len(x_dp)):

            # create cutouts from the decay phase
            steps = int(self.fs / 1000 * 50)
            delta_right = int(min(steps, self.signal_lp.shape[0] - (x_dp[i])))
            ydata = self.signal_lp[x_dp[i]:x_dp[i] + delta_right].reshape(-1,)

            if len(ydata) == steps:
                cutouts_decay.append(ydata)

        cutout_decay_mean = np.mean(np.array(cutouts_decay), axis=0)

        # find the values range
        cdm_range = np.max(cutout_decay_mean) - np.min(cutout_decay_mean)

        # normalize the mean trace between 0 and 1
        cdm_scaled = (cutout_decay_mean - np.min(cutout_decay_mean)
                      ) / cdm_range

        # get datapoints number (dp) within 10-90 %
        found_90 = False
        dp_10 = None
        dp_90 = None
        dp_idx = 0
        for _ in range(len(cdm_scaled) - 1):
            if (cdm_scaled[dp_idx] < 0.1):
                if cdm_scaled[dp_idx + 1] >= 0.1:
                    dp_10 = dp_idx + 1
            if cdm_scaled[dp_idx] < 0.9:
                if (cdm_scaled[dp_idx + 1] >= 0.9) and (not found_90):
                    dp_90 = dp_idx + 1
                    found_90 = True
            dp_idx += 1

        try:
            decay_time_dp = dp_90 - dp_10

            # calculate the rise time (10-90 %)
            decay_time = decay_time_dp / (self.fs / 1000)

        except TypeError:
            decay_time = np.nan

        return decay_time, dp_10, dp_90
