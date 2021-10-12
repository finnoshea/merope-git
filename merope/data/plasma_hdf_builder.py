# turns the pickle data into a single HDF5 file

import os

import numpy as np
import pickle
import h5py


class HDFMaker:
    """
    """
    def __init__(self, directory, shot_number):
        """
        Opens files of the form SHOT###ECE and SHOT###BES and combines them
        into an HDF5 files.

        Parameters
        ----------
        directory : str
            Path to the files.
        shot_number : str
            Shot # to open.
        """
        self.directory = self.dirCheck(directory)
        self.shot_number = self.shotCheck(shot_number)
        self.files = self.fileCheck(self.shot_number)

    @staticmethod
    def dirCheck(directory):
        """
        Checks to make sure a directory is valid.

        Parameters
        ----------
        directory : str
            Directory path.

        Returns
        -------
        If the directory is valid, return it, else None.
        """
        if os.path.isdir(directory):
            return directory
        out = '{:s} does not appear to be a valid directory.'.format(directory)
        raise NotADirectoryError(out)

    @staticmethod
    def shotCheck(shot_number):
        """
        Makes sure that shot number is a string.

        Parameters
        ----------
        shot_number : int or string
            A shot number.

        Returns
        -------
        str of the shot number
        """
        if isinstance(shot_number, int):
            return str(shot_number)
        elif isinstance(shot_number, str):
            return shot_number
        raise TypeError('Please pass shot_number as int or str.')

    def fileCheck(self, shot_number):
        """
        Check which files exist for a given shot number.
        Parameters
        ----------
        shot_number : str
            Shot number to convert to HDF5

        Returns
        -------
        dict of file type : filename
        """
        dd = {}
        suffixes = ['ECE', 'BES']
        for suffix in suffixes:
            fn = os.path.join(self.directory, shot_number + suffix)
            if os.path.isfile(fn):
                dd[suffix] = fn
        if len(dd) > 0:
            return dd
        out = 'Shot number {:s} does not appear to exist.'.format(shot_number)
        raise FileNotFoundError(out)

    @staticmethod
    def openFile(filename):
        """
        Opens a pickle file.

        Parameters
        ----------
        filename : str
            Full path and filename for file to be opened.

        Returns
        -------
        The unpickled data from the file.
        """
        try:
            with open(filename, 'rb') as afile:
                return pickle.load(afile)
        except FileNotFoundError:
            return None

    def convertFiles(self, target):
        """
        Parses the files found in self.files and stores them as a joint HDF5.

        Parameters
        ----------
        target : str
            Full path and filename for HDF5 file to be created.
        """
        data = {}
        for key, path in self.files.items():
            data[key] = self.openFile(path)

        with h5py.File(target, 'w') as hdf:
            pass
        