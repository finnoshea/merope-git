# module for testing converting the csv data into other formats

import os

import numpy as np
import pandas as pd
import dask.dataframe as dd

class APSRunCollector:
    """
    """
    def __init__(self,directory):
        """
        Advanced Photon Source Run Constructor

        Builds a single pandas dataframe from all the different runs in a single
        run directory.

        The data is assumed to be organized in the following way:
        basedir/Run20XY-Z/ -> several high-level files
        Many directories like this:
        Run20XY-Z/2019-02-08-02:42:00 <- run that ended in trip
        Run20XY-Z/2019-02-15-02:59:00 <- another tripped run
        ...
        Run20XY-Z/referenceData <- run from similar time period without trip

        This class expects a directory string that goes all the way to a single
        run. i.e. Run20XY-Z level.

        Parameters
        ----------
        directory : string
            The absolute path to a single run from APS.  This directory should
            have several subdirectories in it as described above.
        """
        self.dir = directory
        self.run_name = os.path.split(self.dir)[-1]
        self.df = None

        self.scen_dirs, self.bad_mags = self.findScenarios()


    def findScenarios(self):
        """
        Creates a list of all the scenario directories and the associated bad
        magnets.

        Returns
        -------
        A list of the scenario directories and a list of the bad magnet in each
        scenario
        """
        dirs = []
        bms = []
        logfile = "/sdf/group/ml/datasets/bes_anomaly_data/" \
                   "PSTrips-2020-01-13/collectLog.csv"
        lf = pd.read_csv(logfile) # collectLog.csv dataframe - its easy

        for dir in os.listdir(self.dir):
            full_path = os.path.join(self.dir,dir)
            if os.path.isdir(full_path):
                dirs.append(full_path) # append the path to dirs variable
                dirname = '/'.join([self.run_name,dir])
                bad_mag = lf[lf['DirName'] == dirname]['PSName'].values[0]
                bms.append(bad_mag)
        return dirs,bms


    def collateData(self):
        """
        Collects all the data from a single run and puts it all into one pandas
        dataframe.
        """
        dfs = []

        for scen_dir,bm in zip(self.scen_dirs,self.bad_mags):
            scenario = APSScenarioCollector(scen_dir,bm)
            dfs.append(scenario.collateData()) # the appended are dask dfs
        full_df = dd.multi.concat(dfs,axis='columns')
        return full_df

class APSScenarioCollector:
    """
    """
    def __init__(self,directory,bad_magnet='None'):
        """
        Advanced Photon Source Scenario Collector

        This class is used to collect all the data in a single APS directory.
        The directory is expected to consist of many CSV files.

        The data is assumed to be organized in the following way:
        basedir/Run20XY-Z/ -> several high-level files
        Many directories like this:
        Run20XY-Z/2019-02-08-02:42:00 <- run that ended in trip
        Run20XY-Z/2019-02-15-02:59:00 <- another tripped run
        ...
        Run20XY-Z/referenceData <- run from similar time period without trip

        This class expects a directory string that goes all the way to a single
        scenario.

        Parameters
        ----------
        directory : string
            The absolute path to a single run from APS.  This directory should
            have several subdirectories in it as described above.
        bad_magnet : string
            The sector:name of the magnet that is bad.  If there was no bad
            magnet, uses 'None'
        """
        self.dir = directory
        self.bad_magnet = bad_magnet
        self.df = None # might be used in the future, not storing the data now


    def __str__(self):
        short_string = '/'.join(self.dir.split(os.sep)[-2:])
        return short_string


    def __repr__(self):
        return "APSScenarioCollector({:s})".format(self.dir)


    def collateData(self):
        """
        Opens all of the CSV files in the scenario directory and puts them
        together into one pandas dataframe.

        Returns
        -------
        pandas dataframe
        """
        keys = []
        dfs = []
        names = ["sector","magnet","feature","units"]
        for fn in os.listdir(self.dir):
            if fn.split('.')[-1] == 'csv': # only csv files
                csv_path = os.path.join(self.dir,fn)
                df = self.read_csv(csv_path)
                dfs.append(dd.from_pandas(df,chunksize=10000)) # convert to dask
        full_df = dd.multi.concat(dfs,axis='columns')
        return full_df


    def collateToSelf(self):
        """
        Opens all of the CSV files in the scenario directory and puts them
        together into one pandas dataframe and then sets self.df to that
        dataframe.
        """
        self.df = self.collateData()


    def read_csv(self,filename):
        """
        Reads in an APS CSV file using the built-in pandas method.

        APS csv files have two row headers: feature name and then units

        Parameters
        ----------
        filename: string
            Full path and name of the file.

        Returns
        -------
        a tuple of (sector_name,magnet_name) and the pandas dataframe
        """
        converters={
                    #'Time': np.int32, # this doesn't help, still 32 bits!
                    'CAerrors': np.int8, # bool is the same size
                   }
        df = pd.read_csv(filename,
                         header=[0,1],
                         dtype=np.float32,
                         converters=converters
                        )
        df.columns = self.MultiColumnIndex(df)
        return df


    def returnScenarioInfo(self):
        """
        Returns the run,scenario, and bad_magnet names as determined by the
        directory path.

        Also does some character replacement to comply with pandas oddness.
        """
        def cheapProcess(x):
            return x.replace(':','_').replace('-','_')

        run, scenario = self.dir.split(os.sep)[-2:]

        run = cheapProcess(run)
        scenario = cheapProcess(run)
        bm = cheapProcess(self.bad_magnet)

        return run,scenario,bm


    def MultiColumnIndex(self,df):
        """
        Creates a new MultiIndex of the columns of the given dataframe so that
        several files can be smashed together in a sane manner.

        Old-style multi-index creator, use MultiColumnIndex instead

        Parameters
        ----------
        df : pandas dataframe
            Must be of the type created by self.read_csv

        Returns
        -------
        a tuple of (sector_name,magnet_name) and the MultiIndex
        """
        cols = df.columns.values
        names = []
        units = []
        for x in cols:
            if x[0] == 'CAerrors': # special case
                units.append('None')
            else:
                units.append(x[1])
            # I can't find a way around the repeating that follows
            nome = x[0].split(':')
            if len(nome) == 3:
                sector = nome[0]
                magnet = nome[1]
                names.append(nome[2])
            else:
                names.append(x[0])
        n_cols = len(cols)

        run,scenario,bm = self.returnScenarioInfo()

        ci = zip(*[[run]*n_cols,
                   [scenario]*n_cols,
                   [bm]*n_cols,
                   [sector]*n_cols,
                   [magnet]*n_cols,
                   names,
                   units])
        mi = pd.MultiIndex.from_tuples(list(ci))

        return mi


    def MultiColumnConstructor(self,df):
        """
        Creates a new MultiIndex of the columns of the given dataframe so that
        several files can be smashed together in a sane manner.

        Old-style multi-index creator, use MultiColumnIndex instead.

        Parameters
        ----------
        df : pandas dataframe
            Must be of the type created by self.read_csv
        """
        cols = df.columns.values
        names = []
        units = []
        for x in cols: # why must this be so verbose?
            if x[0] == 'CAerrors': # special case
                units.append('None')
            else:
                units.append(x[1])
            # I can't find a way around the repeating that follows
            nome = x[0].split(':')
            if len(nome) == 3:
                sector = nome[0]
                magnet = nome[1]
                names.append(nome[2])
            else:
                names.append(x[0])
        n_columns = len(cols)
        arrays = [[self.run_name]*n_columns,
                  [self.bad_mag]*n_columns,
                  [sector]*n_columns,
                  [magnet]*n_columns,
                  names,
                  units]
        index_names = ["Run","BadMag","Sector","Magnet","Feature","Units"]
        new_column_names = pd.MultiIndex.from_arrays(arrays,names=index_names)
        return new_column_names
