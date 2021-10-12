# go through the pickle data and maybe save it to h5py format

import os
import numpy as np
import pickle
import h5py

directory = "/gpfs/slac/staas/fs1/g/coffee_group/edgeml_fes_data/ecebes"


def getData(filename):
    """
    Returns the data from a pickle file.

    Parameters
    ----------
    filename : str
        The name of a file.

    Returns
    -------
    The pickle data in dictionary form.
    """
    with open(filename, 'rb') as afile:
        return pickle.load(afile)


def walkPickle(ds, name):
    """
    Prints out the file structure of a pickle file.

    Parameters
    ----------
    ds : pickle element
        The thing to print out
    name : str
            The name of the element
    """

    def recursiveWalk(elem, name, level):
        """
        Recursive walk of a pickle file.

        Parameters
        ----------
        elem : pickle element
            The thing to print out
        name : str
            The name of the element
        level : int
            The level of the element: higher is deeper, starts at 0.
        """
        post_fix = ''
        if isinstance(elem, np.ndarray):
            post_fix = ' : ' + str(len(elem))
        print('-' * level + '>  ' + name + post_fix)

        if isinstance(elem, dict):
            for key, value in elem.items():
                new_name = name + '/' + key
                recursiveWalk(value, new_name, level + 1)

    return recursiveWalk(ds, name, 0)
