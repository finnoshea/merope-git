# cheap handler for the APS data in CSV format

import os
import csv

import numpy as np
import h5py
import matplotlib.pyplot as plt


class APSDataCollector:
    """
    """
    def __init__(self, directory, filename):
        """
        Object for collecting data from all the CSV files and putting them
        into a single HDF5 file.

        Parameters
        ----------
        directory : str
            Name of a directory to convert into an HDF5 file.
        filename : str
            Name with full path for an H5 file.
        """
        self.directory = directory
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

    def createGroupStructure(self):
        """
        Crawls through self.directory and creates the group structure for the
        HDF5 file that is created in self.filename.
        """

        def recursiveGroupCreation(directory, group):
            """ recursive creation of HDF5 groups """
            group.attrs['path'] = directory
            items = os.listdir(directory)
            if len(items) > 1000:  # leaf dir (a scenario)
                self.addMagnetData(directory, group)
            for dd in items:
                full_path = os.path.join(directory, dd)
                if os.path.isdir(full_path):
                    new_group = self.createSubGroup(group, dd)
                    recursiveGroupCreation(full_path, new_group)

        with h5py.File(self.filename, 'a') as hdfile:
            recursiveGroupCreation(self.directory, hdfile)
            self.addMetaData(self.directory, hdfile)

    def addMagnetData(self, directory, group):
        """
        Collects all the data in a bunch of CSV files in a single directory
        and turns them into datasets.

        Parameters
        ----------
        directory : str
            Path to the CSV files to read.
        group : h5py group
            Location in the h5py file to put the datasets
        """
        for fn in os.listdir(directory):
            if self.checkIfMagnetFile(fn):
                full_path = os.path.join(directory, fn)
                sector_name, magnet_name = fn.split(':')
                magnet_name = magnet_name.split('.')[0]  # drop the .csv
                sector = self.createSubGroup(group, sector_name)
                magnet = self.createSubGroup(sector, magnet_name)
                self.parseMagnetFile(magnet, full_path)

    @staticmethod
    def addMetaData(directory, group):
        """
        Adds the meta data from the file collectLog.csv to the HDF5 file.

        Parameters
        ----------
        directory : str
            Path to the root data directory that contains collectLog.csv
        group : h5py group
            Location in the h5py file to put the datasets
        """
        # rename the labels
        name_dict = {'Run': 'run',
                     'DirName': 'dirName',
                     'PSName': 'PSName',
                     'endDate': 'endDate',
                     'endTimeText': 'endTime',
                     'startDate': 'startDate',
                     'startTimeText': 'startTime'}
        with open(os.path.join(directory, 'collectLog.csv'), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            labels = next(csvreader)
            _ = next(csvreader)  # emtpy but still want to read it
            for row in csvreader:  # have to iterate anyway, do it here
                scenario = group[row[-1].lower()]
                for label, value in zip(labels, row):
                    if label == 'PSName':  # special case
                        try:
                            bad_sector, bad_magnet = value.split(':')
                        except ValueError:  # its 'None', a str
                            bad_sector = 'None'
                            bad_magnet = 'None'
                        scenario.attrs['badSector'] = bad_sector.lower()
                        scenario.attrs['badMagnet'] = bad_magnet.lower()
                    else:
                        label = name_dict[label].lower()
                        scenario.attrs[label] = value

    @staticmethod
    def parseMagnetFile(magnet_group, filename):
        """
        Parses a single magnet csv file.

        Parameters
        ----------
        magnet_group : h5py group
            The h5 group to which to add the data from the file
        filename : str
            Full path and name to the file to process.
        """
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            names = next(csvreader)
            units = next(csvreader)
            data = np.asarray(list(csvreader), dtype='float')
        # iterate over the columns
        for name, unit, column in zip(names, units, data.T):
            # figure out the dtype
            if name == 'CAerrors':
                data_type = 'bool_'  # 8-bit integer
                column = column.astype('bool_')
            elif name == 'Time':
                data_type = '<i8'  # 64-bit integer
                column = column.astype('<i8')
            else:
                data_type = '<f4'  # 32-bit float
            # store it
            name = name.split(':')[-1]
            ds = magnet_group.create_dataset(name.lower(),
                                             data=column,
                                             dtype=data_type)
            ds.attrs['units'] = unit

    @staticmethod
    def checkIfMagnetFile(filename):
        """
        Checks to see if the given filename is consistent with the APS magnet
        CSV file naming convention: SXYA:Q2.csv
        S - always present
        XY - a one- or two-digit number indicating the sector number
        A or B - the sector letter
        :
        Magnet type letter - Q (quad), S (sextupole), H (horizontal corrector),
                            V (vertical corrector)

        Parameters
        ----------
        filename : str
            The name to check.

        Returns
        -------
        bool
            True if the name complies with the convention, otherwise False.
        """
        names = filename.split(':')  # must have a colon
        if len(names) == 2:  # must split in two
            sector = names[0]
            if len(sector) in [3, 4]:  # SXYA
                magnet, extension = names[-1].split('.')
                if extension.lower() == 'csv':  # must be csv file
                    if len(magnet) == 2:  # magnet names are all length 2
                        return True
        return False  # anything else is eliminated

    @staticmethod
    def createSubGroup(group, sub_group_name):
        """
        Attempts to create a sub-group within a group.

        Parameters
        ----------
        group : h5py group
            The group to which a sub-group should be added.
        sub_group_name : str
            The name of the new sub-group.

        Returns
        -------
        hdf5 group
            The sub-group.
        """
        sub_group_name = sub_group_name.lower()
        try:
            new_group = group.create_group(sub_group_name)
        except ValueError:  # the group already exists
            new_group = group[sub_group_name]
        return new_group


class APSDataHandler:
    """
    """
    def __init__(self, filename):
        """
        Object for handling data once it is inside an HDF5 file.

        Parameters
        ----------
        filename : str
            Name with full path for an HDF5 file.
        """
        self.filename = filename

    def __enter__(self):
        self._hdf = h5py.File(self.filename, 'r')
        return self._hdf

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._hdf.close()

    def createDataFrame(self, filters=None):
        """
        Uses filters to select data and puts everything into a pandas
        dataframe.

        Parameters
        ----------
        filters : dictionary

        Returns
        -------
        Pandas dataframe.
        """
        # use the context manager here
        pass

    def parseFilters(self, filters=None):
        """
        Parses a dictionary of filters that can then be used to filter the
        stored data.

        Parameters
        ----------
        filters : dictionary

        Returns
        -------
        XXXXXXXX
        """
        pass

    @staticmethod
    def getObjectType(node):
        """
        The data file contains a strict structure to organize the all the
        information.  The structure is
        /run/scenario/sector/magnet/feature
        All elements are groups except the last, which is a dataset.

        This function returns that type based on the name of the object in
        the dataset.

        Ex:
        node.name = '/fizz/buzz' is a scenario

        Parameters
        ----------
        node : h5py object
            The object to type.

        Returns
        -------
        str
            A string naming the type.  Possibly 'unknown.'
        """
        name = node.name
        if len(node.name) == 1:  # it is the base file, i.e. '/'
            return 'file'
        length = len(name.split('/'))
        names = {2: 'scenario',
                 3: 'sector',
                 4: 'magnet',
                 5: 'feature'}
        try:
            return names[length]
        except KeyError:  # deeper then allowed
            return 'unknown'

    @staticmethod
    def getFeature(hdf, run, scenario, sector, magnet, feature):
        """
        Returns the feature and units for a specific magnet and feature.

        Parameters
        ----------
        hdf : h5py file pointer
            The data to access.
        run : str
            The run to access.
        scenario : str
            The scenario to access.
        sector : str
            The sector to access.
        magnet : str
            The magnet to access.
        feature : str
            The feature to access.

        Returns
        -------
        dict
            A dictionary containing: {units: <the units>,series: <the data>}.
            If an invalid feature is requested, the dictionary will be empty.
        """
        dd = {}
        path = '/' + '/'.join([run, scenario, sector, magnet]).lower()
        try:  # separate addressing and feature access for ease of debugging
            magnet_grp = hdf[path]
        except KeyError as kerr:
            print(kerr)
            print('Check that {:s} all exist.'.format(path))
            return
        try:
            feature = feature.lower()
            dd['units'] = magnet_grp[feature].attrs['units']
            dd['series'] = magnet_grp[feature][()]
        except KeyError as kerr:
            print(kerr)
            print('Check that feature {:s} exists.'.format(feature))
        return dd

    def getMagnetFeatures(self, hdf, run, scenario, sector, magnet):
        """
        Returns the features and units for a specific magnet.

        This method also converts the time from UNIX seconds to a larger set of
        feature vectors as determined by self.processTime.

        Parameters
        ----------
        hdf : h5py file pointer
            The data to access.
        run : str
            The run to access.
        scenario : str
            The scenario to access.
        sector : str
            The sector to access.
        magnet : str
            The magnet to access.

        Returns
        -------
        dict
            A dictionary of dictionaries of the form with keys 'units' and
            'series' for each feature.
        """
        dd = {}
        path = '/' + '/'.join([run, scenario, sector, magnet]).lower()
        for feature in hdf[path].keys():
            td = self.getFeature(hdf, run, scenario, sector, magnet, feature)
            if feature != 'time':  # all other features get added directly
                dd[feature] = td
            else:
                td = self.processTime(td['series'])
                for key, value in td.items():
                    dd[key] = value
        return dd

    @staticmethod
    def processTime(time_series):
        """
        Converts a time series of UNIX seconds into a pair of series
        representing:
        day of the year
        second of the day

        Parameters
        ----------
        time_series : ndarray
            An ndarray of increasing integers representing the UNIX time

        Returns
        -------
        dict
            A dictionary of key:value pairs.
        """
        dd = {}
        # convert to numpy datetime
        time_series = time_series.astype('datetime64[s]')
        day = ((time_series - time_series.astype('datetime64[Y]')) /
               np.timedelta64(1, 'D')).astype(int)
        dd['day'] = {'units': 'days', 'series': day}
        second = (time_series -
                  time_series.astype('datetime64[D]')).astype(int)
        dd['second'] = {'units': 'seconds', 'series': second}
        return dd

    def _listScenarios(self):
        """
        Creates a list of all available scenarios.

        Returns
        -------
        list
            A list of the all the runs and scenarios.
        """
        the_list = []

        with h5py.File(self.filename, 'r') as hdf:
            for runkey, run in hdf.items():  # list all the runs
                the_list.append(runkey)
                for scenkey, scen in run.items():  # list all the scenarios
                    the_list.append('/'.join([runkey, scenkey]))
        return the_list

    def printScenarios(self):
        """
        Prints out a list with all available runs and scenarios.
        """
        the_list = self._listScenarios()
        for elem in the_list:
            if len(elem.split('/')) > 1:
                print('  --> {:s}'.format(elem))
            else:  # run name
                print('+ {:s}'.format(elem))

    def plotBadVersusGood(self, scenario):
        """
        Plots the time series for the scenario given (assumed pathological)
        versus the time series from the associated reference scenario for the
        magnet that caused the trip.

        Parameters
        ----------
        scenario : str
            A path to a scenario in the form run/scenario-name
        """
        try:
            run, scen = scenario.split('/')
        except ValueError as terr:
            print(terr)
            print(('Check that your scenario path format has only one / \n'
                   ' between the run name and scenario name.'))
            return

        if scen == 'referencedata':  # user gave a referencedata directory
            print(('You passed {:s}, which is a '
                   'reference scenario, \n'
                   ' please pass a pathological scenario'.format(scenario)))
            return

        with h5py.File(self.filename, 'r') as hdf:
            badsec = hdf[scenario].attrs['badSector']
            badmag = hdf[scenario].attrs['badMagnet']
            scen = self.getMagnetFeatures(hdf,
                                          run,
                                          scen,
                                          badsec,
                                          badmag)
            ref = self.getMagnetFeatures(hdf,
                                         run,
                                         'referencedata',
                                         badsec,
                                         badmag)

            if len(scen.keys()) != len(ref.keys()):
                print('The scenario and the reference do not have the same'
                      'number of features.  Why?')
                return

            nfeats = len(scen.keys())
            ncols = 3
            nrows = np.ceil(nfeats / ncols).astype('int')
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                    figsize=(2*ncols, 2*nrows),
                                    squeeze=True)
            for idx, key in enumerate(scen.keys()):
                row = idx // ncols
                col = idx % ncols
                ax = axs[(row, col)]
                ax.plot(scen[key]['series'])
                ax.plot(ref[key]['series'])
                ax.set_xlabel(key)
                ax.set_xticks([])
                ax.set_yticks([])
            # delete the unused axes
            if idx < nrows*ncols - 1:
                for x in range(idx+1, nrows*ncols):
                    row = x // ncols
                    col = x % ncols
                    axs[(row, col)].set_visible(False)

            plt.suptitle("{:s} | {}:{}\n blue = bad, "
                         "orange = good".format(scenario, badsec, badmag))
            plt.show()

    def histBadAndNormal(self, scenario, feature):
        """
        Plots a histograms for the scenario given (assumed pathological)
        and the reference scenario for the
        magnet that caused the trip.

        Parameters
        ----------
        scenario : str
            A path to a scenario in the form run/scenario-name
        feature : str
            The magnet feature to histogram.
        """
        try:
            run, sce = scenario.split('/')
        except ValueError as terr:
            print(terr)
            print(('Check that your scenario path format has only one / \n'
                   ' between the run name and scenario name.'))
            return

        if sce == 'referencedata':  # user gave a referencedata directory
            print(('You passed {:s}, which is a '
                   'reference scenario, \n'
                   ' please pass a pathological scenario'.format(sce)))
            return

        with h5py.File(self.filename, 'r') as hdf:
            badsec = hdf[scenario].attrs['badSector']
            badmag = hdf[scenario].attrs['badMagnet']
            scen = self.getFeature(hdf,
                                   run,
                                   sce,
                                   badsec,
                                   badmag,
                                   feature)
            ref = self.getFeature(hdf,
                                  run,
                                  'referencedata',
                                  badsec,
                                  badmag,
                                  feature)
            for key in hdf[run].keys():
                if key in [sce, 'referencedata']:
                    pass  # do nothing, you already have it
                else:
                    new_data = self.getFeature(hdf,
                                               run,
                                               key,
                                               badsec,
                                               badmag,
                                               feature)
                    bs = hdf[run][key].attrs['badSector']
                    bm = hdf[run][key].attrs['badMagnet']
                    if bs == badsec and bm == badmag:  # similar pathology
                        scen['series'] = np.append(scen['series'],
                                                   new_data['series'])
                    else:  # closer to reference
                        ref['series'] = np.append(ref['series'],
                                                  new_data['series'])
        badstr = 'bad, N = {:d}'.format(len(scen['series']))
        norstr = 'normal, N = {:d}'.format(len(ref['series']))
        plt.hist(scen['series'], bins=50, label=badstr)
        plt.hist(ref['series'], bins=50, label=norstr)
        plt.legend(loc='upper left')
        plt.title('{:s} - {:s}:{:s} - {:s}'.format(scenario,
                                                   badsec, badmag, feature))
        plt.show()
