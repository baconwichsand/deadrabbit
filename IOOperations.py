import csv
import h5py
import time
import numpy as np
from ConfigParser import *


class HDF5:

    #
    # Construct HDF5 operator
    #
    # Modes:
    #    -- 'a': read/write if exists, create otherwise
    #    -- 'r': read only, file must exist
    #    -- 'r+': read/write, file must exist
    #
    def __init__(self, filename, mode):
        self.operator = h5py.File(filename, mode)

    #
    # End HDF5 access
    #
    def end_access(self):
        self.operator.close()

    #
    # Add data to group given data, data name, and group path
    #
    def add_dset(self, dset, dset_name, group_path=''):
        self.operator[group_path + '/' + dset_name] = dset

    #
    # Retrieve dataset in database
    #
    def get_dset(self, dset_name, group_path=''):
        dset = self.operator[group_path + '/' + dset_name]
        return dset[...]

    #
    # Get subset of dataset in database
    #
    def get_dset_subset(self, dset_name, start_index, end_index, group_path=''):
        dset = self.operator[group_path + '/' + dset_name]
        return dset[...][start_index:end_index]


    #
    # Returns array of all values in a group
    #
    def get_group(self, group_name, group_path=''):
        output = []
        for i, x in enumerate(self.operator[group_path + '/' + group_name].itervalues()):
            output.append(x[...])
        return output

    #
    # Add attribute to dataset in database
    #
    def add_attr(self, dset_name, attribute_name, attribute_value, group_path=''):
        dset = self.operator[group_path + '/' + dset_name]
        dset.attrs[attribute_name] = attribute_value

    #
    # Retrieve attribute from dataset in database
    #
    def get_attr(self, dset_name, attribute_name, group_path=''):
        dset = self.operator[group_path + '/' + dset_name]
        return dset.attrs.get(attribute_name)

    #
    # Create session
    #
    def new_session(self):
        t = time.strftime('%c')
        self.operator.create_group(t)
        return t


class Config:

    #
    # Creates Config object
    #
    def __init__(self, filename):
        self.operator = ConfigParser()
        self.filename = filename

    #
    # Return specific parameter
    #
    def get(self, section, option, dtype=str):
        self.operator.readfp(open(self.filename, 'r'))
        return dtype(self.operator.get(section, option))

    #
    # Returns an entire section values as array of strings
    #
    def get_section(self, section):
        self.operator.readfp(open(self.filename, 'r'))
        output = []
        for x in self.operator.items(section):
            output.append(x[1])
        return output


class CSV:

    #
    # Creates CSV object
    #
    def __init__(self, filename):
        self.filename = filename

    #
    # Read column in csv to array
    #
    # Optional parameter: dtype, parse to specific datatype
    #
    def read_csv_rows(self, dtype=float):
        output = []
        f = open(self.filename, 'rt')
        reader = csv.reader(f)
        for row in reader:
            output.append(row)
        f.close()
        output = np.array(output).astype(dtype)
        return output[:, 1]
