# explore the BNL data cheaply

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

data_path = '/sdf/group/ml/datasets/bes_anomaly_data/ML_NSLS2_Cryo'
filenames = ['cry_2021-1-21.hdf5',
             'cry_2021-4-30.hdf5',
             'cry_2021-9-03.hdf5']


def plot_data(filename):
    with h5py.File(os.path.join(data_path, filename), 'r') as hdf:
        fig, ax = plt.subplots()
        times = hdf['SR-Cry{CBox}T:3605-I'][1, :]
        time_0 = times[0]
        times -= time_0
        times /= 3600  # time in hours
        signal = hdf['SR-Cry{CBox}T:3605-I'][0, :]
        ax.plot(times, signal)
        status = hdf['SR-OPS{}Mode-Sts'][0, :]
        status_times = hdf['SR-OPS{}Mode-Sts'][1, :]
        for idx, flag in enumerate(((status == 5) | (status == 6))[:-1]):
            if flag:
                ax.axvspan((status_times[idx] - time_0) / 3600,
                           (status_times[idx + 1] - time_0) / 3600,
                           alpha=0.5,
                           color='red')
        plt.xlabel('time (hours)')
        plt.ylabel('SR-Cry{CBox}T:3605-I')
        plt.title(filename)
        plt.show()