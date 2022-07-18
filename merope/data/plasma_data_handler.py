# handles HDF5 files of plasma runs

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal


class PlasmaDataHandler:
    """
    """
    def __init__(self, filename):
        """
        Class to manage interaction with the HDF5 files created by HDFMaker.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to open.
        """
        self.filename = self.fileCheck(filename)

    @staticmethod
    def fileCheck(filename):
        """
        Check whether the given file exists.

        Parameters
        ----------
        filename : str
            The file to look for

        Returns
        -------
        dict of file type : filename
        """
        if os.path.isfile(filename):
            return filename
        out = "File {:s} does not exist.".format(filename)
        raise FileNotFoundError(out)

    def listInstruments(self, printout=True):
        """
        Creates a description of the available instruments and channels in
        the handled HDF5 file.

        Parameters
        ----------
        printout : bool
            If True, prints out available instruments and channels.

        Returns
        -------
        If printout=False, returns a dictionary.
        """
        dd = {}
        with h5py.File(self.filename, 'r') as hdf:
            for key, value in hdf.items():
                v_keys = sorted(list(value.keys()))[:-1]  # remove 'time'
                dd[key] = v_keys
        if printout:
            for k, v in dd.items():
                print('Instrument: {:s}'.format(k))
                print('Channels: ', v)
            return
        return dd

    def returnShotNumber(self):
        """
        Returns the integer shot number.
        """
        with h5py.File(self.filename, 'r') as hdf:
            return hdf.attrs['shot_number']

    def returnCoincidentData(self, instrument, channel):
        """
        Returns the requested data in the coincident window.

        Parameters
        ----------
        instrument : string
            Instrument to use: 'ECE' or 'BES'
        channel : int
            Channel number to use.  Varies by instrument.

        Returns
        -------
        ndarray of data, float of time step
        """
        with h5py.File(self.filename, 'r') as hdf:
            hdf_path = '/'.join([instrument, "{:0>2}".format(channel)])
            try:
                ci = hdf[instrument].attrs['coincident_indexes']
                ts = hdf[instrument].attrs['time_step']
                data = hdf[hdf_path][ci[0]:ci[-1]]  # get the coincident dataset
            except KeyError as err:
                print(err)
                print('Check that the instrument and channel both exist.')
                return
        return data, ts

    def buildFilter(self, window_size, n_windows, length=50):
        """
        Builds a rolling-average filter.

        Parameters
        ----------
        window_size : int
            Size of the windows used to compute the FFT.
        n_windows : int
            The number of windows in the spectrogram
        length : int
            The length of the rolling average.

        Returns
        -------
        ndarray
        """
        return np.ones((n_windows, length)) / length

    def yieldInstrumentChannels(self, instrument):
        """
        Yields all the channel numbers in an instrument.

        Parameters
        ----------
        instrument : str
            ECE or BES

        Yields
        -------
        ints representing the keys.
        """
        with h5py.File(self.filename, 'r') as hdf:
            keys = list(hdf[instrument].keys())
        for key in keys:
            try:
                emit = int(key)
            except ValueError:  # not a channel, likely 'time'
                continue
            yield emit

    def yieldAllInstrumentChannels(self, instrument):
        """
        Yields all the coincident channel data for an instrument.

        Parameters
        ----------
        instrument : str
            ECE or BES

        Yields
        -------
        Channel number (int), coincident data (ndarray), and coincident times
        for each channel
        """
        with h5py.File(self.filename, 'r') as hdf:
            keys = list(hdf[instrument].keys())
            ci = hdf[instrument].attrs['coincident_indexes']
            ts = hdf[instrument].attrs['time_step']
            for key in keys:
                hdf_path = '/'.join([instrument, "{:0>2}".format(key)])
                data = hdf[hdf_path][ci[0]:ci[-1]]  # the coincident dataset
                try:
                    emit = int(key)
                except ValueError:  # not a channel, likely 'time'
                    continue
                yield emit, data, ts

    @staticmethod
    def scaleMinMax(minimum, maximum, scale):
        """
        Rescales minimum and maximum to be scale times smaller.  If minimum is
        positive it is not changed.  Used for adjusting colormaps.

        Parameters
        ----------
        minimum : float
            The minimum value in the colormap.
        maximum : float
            The maximum value in the colormap.
        scale : float
            The scale to change to.  (0,1.0]

        Returns
        -------
        Two floats, vmin and vmax for matplotlib's imshow
        """
        if maximum == minimum:  # all zeros, I guess.
            return 0, 1  # arbitrary
        if maximum <= 0 or minimum >= 0:
            return minimum, scale * maximum
        # should be max > 0 and min < 0 now
        return scale * minimum, scale * maximum

    def createSpectrogram(self, instrument,
                          channel, window_size,
                          make_plot=True,
                          scale=1.0):
        """
        Creates a spectrogram of the given instrument and channel.

        Parameters
        ----------
        instrument : string
            Instrument to use: 'ECE' or 'BES'
        channel : int
            Channel number to use.  Varies by instrument.
        window_size : int
            Size of the windows used to compute the FFT.
        make_plot : bool
            If True, produces a matplotlib plot, else returns the spectrogram.
        scale : float
            Change the scale of the colors from full (1.0), to some fraction.

        Returns
        -------
        If make_plot=False, returns an ndarray.
        """
        x, ts = self.returnCoincidentData(instrument, channel)
        data_length = len(x)
        n_windows = data_length // window_size  # the number of windows
        # rows are windows - for computational efficiency
        x = x[:n_windows * window_size].reshape(n_windows, window_size)
        amp = np.abs(np.fft.fft(x, axis=1))
        if np.max(amp) == 0:
            return
        out = np.log(amp + 1e-12)  # add a small number to avoid -inf in log
        filt = self.buildFilter(window_size, n_windows, 50)
        bg = signal.fftconvolve(out, filt, mode='same', axes=1)
        out -= bg
        out[:, 0] = 0
        out -= np.mean(out)
        out *= 1 / np.std(out)

        # remove the redundant frequencies
        out = out[:, :window_size // 2].T

        if make_plot:
            # plot the spectrogram
            wsize = 12
            #hsize = wsize * window_size / (2 * n_windows)
            hsize = 4
            fig, ax = plt.subplots(figsize=(wsize, hsize))

            vmin, vmax = self.scaleMinMax(np.min(out), np.max(out), scale)
            ax.imshow(out, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)

            ax.set_xticks(np.linspace(0, n_windows, 5, endpoint=True))
            ax.set_xticklabels(np.linspace(0, n_windows * window_size, 5,
                                           endpoint=True) * ts / 1000.0)
            ax.set_xlabel(r'time (ms)')
            ax.set_yticks(np.linspace(0,
                                      window_size // 2,
                                      5,
                                      endpoint=True)
                          )

            max_idx = (window_size - 1) // 2
            max_freq = np.fft.fftfreq(window_size, d=ts * 1e-3)[max_idx]  # kHz
            yticks = ['{:.1f}'.format(tick) for tick in
                      np.linspace(0, max_freq, 5, endpoint=True)]
            ax.set_yticklabels(yticks)
            ax.set_ylabel('f (kHz)')
            shot_number = self.returnShotNumber()
            title_str = "Shot {:d} : {:s}, channel {:0>2d}".format(shot_number,
                                                                   instrument,
                                                                   channel)
            ax.set_title(title_str)
            plt.show()
        else:
            return out
