# run the MSR algorithm on FERMI data

import os

import numpy as np
import scipy.linalg as linalg
import h5py
import matplotlib.pyplot as plt

filedir = '/sdf/group/ml/datasets/bes_anomaly_data/ML_NSLS2_Utils'
filepointer = filedir + '/2020_Apr_1-20.h5'

failtime = 1586273400  # UNIX timestamp

def MSRdata(timeseries, m=60):
    """
    Runs the Modified Spectral Residual Algorithm by Ryan Humble on an input
    timeseries

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------

    """
    alp = np.zeros((m, m))  # historical AL matrix
    sn = np.zeros(len(timeseries), dtype=complex)  # Sn vector
    nn = np.zeros(len(timeseries), dtype=int)    # track the number of entries

    def updatealp(old, new):
        """
        Updates the average log amplitude (ALP) matrix.

        Parameters
        ----------
        old : ndarray
            The previous ALP matrix
        new : ndarray
            1D array to append to the old ALP matrix.

        Returns
        -------
        ndarray
            The updated ALP matrix
        """
        return np.vstack((old[1:, :], new))

    idx = m
    U = linalg.dft(m)
    invU = linalg.inv(U)
    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        x -= np.mean(x)
        X = np.matmul(U, x)                 # step 1
        X[0] = 1.  # kludge
        A = np.abs(X)                       # step 2
        P = (0 - 1j) * np.log(X / A)        # step 3
        L = np.log(A)                       # step 4
        AL = np.mean(alp, axis=0)           # step 5
        alp = updatealp(alp, L)
        RA = L - AL                         # step 6
        R = np.exp(np.abs(RA)) \
            * np.exp(1j * P)                # step 7
        R[0] = 0.  # unkludge
        r = np.matmul(invU, R)              # step 8
        sn[idx-m:idx] += r
        nn[idx-m:idx] += np.ones(m, dtype=int)

        idx += 1  # update iterator

    return np.abs(sn) / nn


def TMNMSR(timeseries, m=60):
    """
    Triple Modified Spectral Residual - but outputting all rs

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------
    ndarrays of Sn and a matrix of all the rs
    """
    alp = np.zeros((m, m))  # historical AL matrix
    sn = np.zeros(len(timeseries), dtype=complex)  # Sn vector
    nn = np.zeros(len(timeseries), dtype=int)    # track the number of entries

    def updatealp(old, new):
        """
        Updates the average log amplitude (ALP) matrix.

        Parameters
        ----------
        old : ndarray
            The previous ALP matrix
        new : ndarray
            1D array to append to the old ALP matrix.

        Returns
        -------
        ndarray
            The updated ALP matrix
        """
        return np.vstack((old[1:, :], new))

    def geo_mean(a):
        """
        Computes the geometric mean.

        Parameters
        ----------
        a : ndarray
            Array with time along the rows and frequency along the columns.
        Returns
        -------
        ndarray of the geometric mean for each frequency
        """
        return np.exp(np.mean(np.log(a), axis=0))

    idx = m
    U = linalg.dft(m)
    invU = linalg.inv(U)
    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        x -= np.mean(x)
        X = np.matmul(U, x)                 # step 1
        AX = geo_mean(alp)                  # step 2
        alp = updatealp(alp, X)
        R = X - AX                          # step 3
        r = np.matmul(invU, R)              # step 4
        sn[idx-m:idx] += r                  # step 5 - almost
        nn[idx-m:idx] += np.ones(m, dtype=int)

        idx += 1  # update iterator

    return np.abs(sn) / nn  # the rest of step 5


def MSRHDFs(algo=MSRdata, filepointer=filepointer, m=60):
    a = 'real time/Bunch Number/BunchNumber'
    #b = 'SR:C08-EPS{FLM:08}Val-I'
    b = 'SR:C05-EPS{FLM:05}Val-I'
    f = filepointer.split('/')[-1] + '  ' + b + '  ' + repr(algo).split()[1]
    datas = np.array([])
    sns = np.array([])
    bns = np.array([])
    with h5py.File(filepointer, 'r') as afile:
        data = afile[b]['value'][()].flatten()  # the ends are being weird
        bn = afile[b]['meta']['sec'].flatten()
        sn = algo(data, m)

        datas = np.hstack((datas, data))
        bns = np.hstack((bns, bn))
        sns = np.hstack((sns, sn))

    # define a threshold
    logsn = np.log(sns)
    logsn = logsn[np.isfinite(logsn)]  # leave off the zeros / infinities
    mean_sn = np.mean(logsn)
    std_sn = np.std(logsn)
    num_dev = 2
    threshold = np.exp(mean_sn + num_dev * std_sn)
    anomalies = sns >= threshold

    fig, axs = plt.subplots(2, 1, squeeze=True, sharex='col')

    axs[0].scatter(bns[~anomalies], datas[~anomalies], s=2, c='b')
    axs[0].scatter(bns[anomalies], datas[anomalies], s=2, c='r')
    axs[0].set_ylabel(b)
    axs[0].set_title(f)

    axs[1].plot([min(bns), max(bns)], [threshold] * 2, '-.k')
    axs[1].semilogy(bns, sns, color='r',
                    marker='.', markersize=1, linestyle='')
    axs[1].set_ylabel('saliency')
    axs[1].set_xlabel('time')
    axs[1].set_title('threshold = {:4.3f}, m = {:d}'.format(threshold, m))

    plt.show()

    return (bns, sns)

