# cheap handler for the BNL data in CSV format

import numpy as np
import h5py
import matplotlib.pyplot as plt


class BNLDataHandler:
    """
    """
    def __init__(self, filename):
        """
        Object for parsing the HDF5 files from BNL and turning them into
        data for use in a machine learning algorithm.

        Parameters
        ----------
        filename : str
            Name with full path for an HDF5 file.
        """
        self.filename = filename
        self.hdf = None  # the HDF5 file

    def openFile(self):
        """
        Opens the HDF5 file in append mode.
        """
        self.hdf = h5py.File(self.filename, 'a')

    def closeFile(self):
        """
        Closes the HDF5 file.
        """
        self.hdf.close()

    def countAllAlarms(self, channel='severity'):
        """
        For the entire file returns a count of the number of channel meta
        variables greater than 0 and a dictionary of PV : count.

        Parameters
        ----------
        channel : str
            'severity' or 'status'
        Returns
        -------
        int and dict
        """
        count = 0
        dd = {}  # store the name of PVs with severity levels > 0
        with h5py.File(self.filename, 'r') as hdffile:
            for key, value in hdffile.items():
                sevs = value['meta'][channel]
                s_count = sum(sevs > 0)
                if s_count > 0:
                    count += s_count
                    dd[key] = s_count
        return count, dd
