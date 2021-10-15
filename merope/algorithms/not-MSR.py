# what is going on inside the various MSR algorithms?

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action="error", category=np.ComplexWarning)


def geo_mean(a):
    """
    Computes the geometric mean of a.

    Parameters
    ----------
    a : ndarray
        Array with time along the rows and frequency along the columns.
    Returns
    -------
    ndarray of the geometric mean for each frequency
    """
    return np.exp(np.mean(np.log(a), axis=0))

def updatehistory(old, new):
    """
    Updates the history matrix.

    Parameters
    ----------
    old : ndarray
        The previous history matrix
    new : ndarray
        1D array to append to the history matrix.

    Returns
    -------
    ndarray
        The updated history matrix
    """
    return np.vstack((old[1:, :], new))

def hermitianInner(a, b):
    """
    The Hermitian inner product of complex vectors a and b.
    conj(a)*b

    Parameters
    ----------
    a : ndarray
        First complex vector.
    b : ndarray
        Second complex vector.

    Returns
    -------
    complex value
    """
    return np.inner(np.conjugate(a), b)


def similarity(a, b):
    """
    Computes the dissimilarity of two complex vectors.

    similarity = Re{<a,b> / (||a|| ||b||)}

    returns |( similarity - 1) / 2|

    Parameters
    ----------
    a : ndarray
        First complex vector.
    b : ndarray
        Second complex vector.

    Returns
    -------
    float
    """
    mag_a = np.sqrt(np.real(hermitianInner(a, a)))
    mag_b = np.sqrt(np.real(hermitianInner(b, b)))
    if mag_a == 0 and mag_b == 0:
        return 0  # they are the same
    if mag_a == 0 or mag_b == 0:
        return 1  # maximally dissimilar
    return np.real(hermitianInner(a, b)) / (mag_a * mag_b)


def creditAssignment(m):
    """
    Create a credit assignment vector.

    Parameters
    ----------
    m : int
        The length of the vector.

    Returns
    -------
    1d ndarray
    """
    vec = np.zeros(m)
    vec[0] = 0.5
    vec[-1] = 0.5
    return vec


def TMNMSR(timeseries, m=10):
    """
    Triple Modified Spectral Residual

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------
    1D ndarray of Sn and 2D ndarray of the complex r values.
    """
    alp = np.zeros((m, m))  # historical AL matrix
    sn = np.zeros(len(timeseries), dtype=complex)  # Sn vector
    nn = np.zeros(len(timeseries), dtype=int)    # track the number of entries
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=complex)

    idx = m
    U = linalg.dft(m)
    invU = linalg.inv(U)
    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        x -= np.mean(x)
        X = np.matmul(U, x)                 # step 1
        AX = geo_mean(alp)                  # step 2
        alp = updatehistory(alp, X)
        R = X - AX                          # step 3
        r = np.matmul(invU, R)              # step 4
        sn[idx-m:idx] += r                  # step 5 - almost
        rs[idx-m:idx, idx-1] = r
        nn[idx-m:idx] += np.ones(m, dtype=int)

        idx += 1  # update iterator
    # the rest of step 5
    return np.abs(sn) / nn, rs


def FourierSimilarity(timeseries, m=10):
    """
    Fourier Similarity of time series windows.

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------
    1D ndarray of similarity coefficients.
    """
    lam = 0.0  # memory parameter - hard code in
    U = linalg.dft(m)
    invU = linalg.inv(U)

    sn = np.zeros(len(timeseries), dtype=float)  # Sn vector
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=float)
    #V = np.zeros(m, dtype=complex)  # history vector

    # cheat on the first m
    sn[:m-1] = np.zeros(m-1)
    rs[:m-1, :m-1] = np.zeros((m-1, m-1))
    V = np.copy(np.matmul(U, timeseries[:m]))

    idx = m

    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        x -= np.mean(x)
        X = np.matmul(U, x)                 # step 1 - DFT
        nu = similarity(X, V)               # step 2a - compute similarity
        nu = np.abs(nu - 1) / 2             # step 2b - compute dissimilarity
        C = X - V
        norm = linalg.norm(C)
        if norm > 0:
            C /= norm
        V *= lam                            # step 3a - forget old Xs
        V += X                              # step 3b - remember new X
        # this is vanilla FS
        # sn[idx-1] += nu
        # rs[idx-1, idx-1] = nu
        # this is equitable FS
        # sn[idx-m:idx] += nu * np.ones(m)
        # rs[idx-m:idx, idx-1] = nu * np.ones(m)
        # this is where I am playing around
        c = np.abs(np.matmul(invU, C))
        sn[idx - m:idx] += nu * c
        rs[idx - m:idx, idx - 1] = nu * c

        idx += 1  # update iterator

    return sn, rs


def FourierDelaySimilarity(timeseries, m=10, delay=1):
    """
    Fourier Similarity of time series windows.

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.
    delay : int
        How far behind the vector V is from vector X

    Returns
    -------
    1D ndarray of similarity coefficients.
    """
    lam = 0.0  # memory parameter - hard code in
    U = linalg.dft(m)
    invU = linalg.inv(U)

    sn = np.zeros(len(timeseries), dtype=float)  # Sn vector
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=float)
    #V = np.zeros(m, dtype=complex)  # history vector

    # cheat on the first m + delay
    V = np.copy(timeseries[:m])

    idx = m + delay

    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        x -= np.mean(x)
        X = np.matmul(U, x)                 # step 1 - DFT
        nu = similarity(X, V)               # step 2a - compute similarity
        nu = np.abs(nu - 1) / 2             # step 2b - compute dissimilarity
        C = X - V
        norm = linalg.norm(C)
        if norm > 0:
            C /= norm
        c = np.abs(np.matmul(invU, C))
        sn[idx - m:idx] += nu * c
        rs[idx - m:idx, idx - 1] = nu * c

        idx += 1  # update iterator

        v = np.copy(
            timeseries[idx - m - delay:idx - delay])  # step 2b - new V
        v -= np.mean(v)
        V = np.matmul(U, v)

    return sn, rs


def FourierSimilarity2(timeseries, m=10):
    """
    Fourier Similarity of time series windows.
    Uses the magnitudes of the vectors to compute dissimilarity.

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------
    1D ndarray of similarity coefficients.
    """
    lam = 0.0  # memory parameter - hard code in
    U = linalg.dft(m)
    invU = linalg.inv(U)

    sn = np.zeros(len(timeseries), dtype=float)  # Sn vector
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=float)
    #V = np.zeros(m, dtype=complex)  # history vector

    # cheat on the first m
    sn[:m-1] = np.zeros(m-1)
    rs[:m-1, :m-1] = np.zeros((m-1, m-1))
    V = np.copy(np.matmul(U, timeseries[:m]))

    idx = m

    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        x -= np.mean(x)
        X = np.matmul(U, x)                 # step 1 - DFT
        nu = similarity(np.abs(X),
                        np.abs(V))          # step 2a - compute similarity
        nu = np.abs(nu - 1) / 2             # step 2b - compute dissimilarity
        C = X - V
        V *= lam                            # step 3a - forget old Xs
        V += X                              # step 3b - remember new X
        c = np.abs(np.matmul(invU, C))
        sn[idx - m:idx] += nu * c
        rs[idx - m:idx, idx - 1] = nu * c

        idx += 1  # update iterator

    return sn, rs


def RealSimilarity(timeseries, m=10):
    """
    Fourier Similarity of time series windows.

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------
    1D ndarray of similarity coefficients.
    """
    lam = 0.0  # memory parameter - hard code in
    U = linalg.dft(m)
    invU = linalg.inv(U)

    sn = np.zeros(len(timeseries), dtype=float)  # Sn vector
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=float)
    #V = np.zeros(m, dtype=complex)  # history vector

    # cheat on the first m
    sn[:m-1] = np.zeros(m-1)
    rs[:m-1, :m-1] = np.zeros((m-1, m-1))
    V = np.copy(timeseries[:m])

    idx = m

    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        X = x - np.mean(x)
        norm = linalg.norm(X)
        if norm > 0:
            X /= norm
        nu = similarity(X, V)               # step 1a - compute similarity
        nu = np.abs(nu - 1) / 2             # step 1b - compute dissimilarity
        C = X - V
        V *= lam                            # step 2a - forget old Xs
        V += X                              # step 2b - remember new X
        sn[idx - m:idx] += nu * np.abs(C)
        rs[idx - m:idx, idx - 1] = nu * np.abs(C)

        idx += 1  # update iterator

    return sn, rs


def RealDelaySimilarity(timeseries, m=10, delay=1):
    """
    Similarity between vectors in real space with delay.

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.
    delay : int
        The size of the delay between the current window and the window
        it is compared with.

    Returns
    -------
    1D ndarray of similarity coefficients.
    """
    lam = 0.0  # memory parameter - hard code in
    U = linalg.dft(m)
    invU = linalg.inv(U)

    sn = np.zeros(len(timeseries), dtype=float)  # Sn vector
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=float)
    #V = np.zeros(m, dtype=complex)  # history vector

    # cheat on the first m + delay
    V = np.copy(timeseries[:m])

    idx = m + delay

    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        X = x - np.mean(x)
        norm = linalg.norm(X)
        if norm > 0:
            X /= norm
        nu = similarity(X, V)               # step 1a - compute similarity
        nu = np.abs(nu - 1) / 2             # step 1b - compute dissimilarity
        C = X - V
        sn[idx - m:idx] += nu * np.abs(C)
        rs[idx - m:idx, idx - 1] = nu * np.abs(C)

        idx += 1  # update iterator

        V *= lam  # step 2a - forget old Xs
        V += np.copy(
            timeseries[idx - m - delay:idx - delay])  # step 2b - new V
        V -= np.mean(V)
        norm_v = linalg.norm(V)
        if norm_v > 0:
            V /= norm_v

    return sn, rs


def SurpriseSimilarity(timeseries, m=10):
    """
    Shannon Surprise Similarity of time series windows.

    Parameters
    ----------
    timeseries : ndarray
        A time series.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------
    1D ndarray of similarity coefficients.
    """
    lam = 0.0  # memory parameter - hard code in
    U = linalg.dft(m)
    invU = linalg.inv(U)

    sn = np.zeros(len(timeseries), dtype=float)  # Sn vector
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=float)
    #V = np.zeros(m, dtype=complex)  # history vector

    # cheat on the first m
    sn[:m-1] = np.zeros(m-1)
    rs[:m-1, :m-1] = np.zeros((m-1, m-1))
    V = np.copy(timeseries[:m])

    idx = m

    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        X = x - np.mean(x)
        norm = linalg.norm(X)
        if norm > 0:
            X /= norm
        nu = (similarity(X, V) + 1) / 2     # step 1a - compute similarity (0<=nu<=1)
        nu = -1*np.log(nu)                  # step 1b - compute surprise
        C = X - V
        norm_c = linalg.norm(C)
        if norm_c > 0:
            C /= norm_c
        V *= lam                            # step 2a - forget old Xs
        V += X                              # step 2b - remember new X
        sn[idx - m:idx] += nu * C
        rs[idx - m:idx, idx - 1] = nu * C

        idx += 1  # update iterator

    return sn, rs

def ShiftSimilarity(timeseries, m=10):
    """
    Fourier Similarity of time series windows.

    Parameters
    ----------
    timeseries : ndarray
        A time series from the FERMI machine.
    m : int
        The size of the window defined in the MSR algorithm.

    Returns
    -------
    1D ndarray of similarity coefficients.
    """
    lam = 0.0  # memory parameter - hard code in
    U = linalg.dft(m)
    invU = linalg.inv(U)

    sn = np.zeros(len(timeseries), dtype=float)  # Sn vector
    rs = np.zeros((len(timeseries), len(timeseries)), dtype=float)
    # V = np.zeros(m, dtype=complex)  # history vector

    # cheat on the first m
    sn[:m - 1] = np.zeros(m - 1)
    rs[:m - 1, :m - 1] = np.zeros((m - 1, m - 1))
    V = np.copy(timeseries[:m])

    idx = m

    while idx <= len(timeseries):
        x = np.copy(timeseries[idx-m:idx])
        X = x - np.mean(x)
        norm = linalg.norm(X)
        if norm > 0:
            X /= norm
        nu = similarity(X, V)               # step 1 - compute similarity
        nu = np.abs(nu - 1) / 2
        C = X - V
        V *= lam                            # step 2a - forget old Xs
        V += np.hstack((X[1:], X[0]))         # step 2b - remember new X
        sn[idx - m:idx] += nu * np.abs(C)
        rs[idx - m:idx, idx - 1] = nu * np.abs(C)

        idx += 1  # update iterator

    return sn, rs


def squareInput(n=20, m=4, s=8):
    """
    Creates a square wave input.
    Parameters
    ----------
    n : int
        The total length of the input
    m : int
        The length of the square wave
    s : int
        The starting location of the square wave

    Returns
    -------
    ndarray of the square wave
    """
    wave = np.zeros(n)
    if m > n:  # don't let the user give a huge range
        m = n
    if s + m > n:  # s is too big, fix it
        s = n - m
    elif s < 0:  # s is too small, fix it
        s = 0
    wave[s:s+m] = np.ones(m) / m
    return wave

def sinusoidInput(n=20, T=8, phase=0):
    """
    Creates a sinusoidal time series of period T with length n.

    Parameters
    ----------
    n : int
        The number of time series samples
    T : float
        The period of the sinusoid.

    Returns
    -------
    ndarray of the sinusoid
    """
    return np.sin(2*np.pi*(np.arange(n) + phase) / T)

def standardNormalNoise(n=20):
    """
    Creates a pure noise signal using a standard normal distribution.

    Parameters
    ----------
    n : int
        The number of time series samples

    Returns
    -------
    ndarray of the time series
    """
    return np.random.standard_normal(n)

def plotAlgo(algo, timeseries, m=10, *args, **kwargs):
    """
    Makes a useful plot of the
    Parameters
    ----------
    algo : MSR-type algorithm that accepts a time series and a window size
        The algorithm to use.
    timeseries : ndarray
        A time series to compute the MSR-type salience
    m : int
        Window size.

    Returns
    -------
    Nothing but a plot.
    """
    Sn, Rn = algo(timeseries, m, *args, **kwargs)

    # start with a square Figure
    fig = plt.figure(figsize=(14, 8))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 3, width_ratios=(7, 7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])  # phase plot
    ax_histx = fig.add_subplot(gs[0, 1], sharex=ax)  # time series plot
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    ax_histy = fig.add_subplot(gs[1, 2], sharey=ax)  # Sn plot
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax, sharey=ax)  # amplitude plot
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax_histx.step(np.arange(len(timeseries)), timeseries, 'b')
    ax_histx.set_ylabel('time series')
    ax_histx.set_title('m = {:d}'.format(m))
    ax_histx.set_xlim([0, len(timeseries) - 1])

    ax.imshow(np.angle(Rn), aspect='auto', cmap='hsv')
    ax.set_xlabel('time step (j)')
    ax.set_ylabel('k')
    ax.text(5, len(timeseries) - 20, r'$arg(r)$',
             fontdict={'color': 'k',
                       'fontsize': 14})

    ax2.imshow(np.abs(Rn), aspect='auto')
    ax2.set_xlabel('time step (j)')
    ax2.text(5, len(timeseries) - 20, r'$|r|$',
            fontdict={'color': 'w',
                      'fontsize': 14})

    ax_histy.plot(Sn, np.arange(len(timeseries)), 'b')
    ax_histy.set_xlabel(r'$S_n$')
    ax_histy.set_ylim([0, len(timeseries)-1])

    plt.show()


standard_sinusoid = sinusoidInput(n=500, T=39.2)
standard_sinusoid[290] += 0.2
standard_sinusoid[360] -= 0.2

phased_sinusoid = sinusoidInput(n=500, T=39.2)
temp = sinusoidInput(n=500, T=39.2, phase=1)
phased_sinusoid[280:] = temp[280:]
