from IOOperations import HDF5, CSV
import rand_generators
from DataManipulation import remove_x
import numpy as np


class InputBuilder:

    #
    # Constructs an InputBuilder that writes to a HDF5 db
    #
    def __init__(self, hdf5file):
        self.hdf5 = hdf5file


class TargetData(InputBuilder):

    #
    # Name of group where TargetData is stored in HDF5 database
    #
    GROUP = '/TargetData'

    #
    # Generate target data from CSV file and add to database
    #
    def csv(self, filename, dset_name):
        r = CSV(filename)
        output = remove_x(r.read_csv_rows())
        hdf5 = HDF5(self.hdf5, 'a')
        hdf5.add_dset(output, dset_name, group_path=TargetData.GROUP)
        hdf5.end_access()
        return np.array(output)

    #
    # Generate target data from random generator
    #
    # Available types:
    #    -- 'volatility': max volatility between float values, params = [start_int, vol_float]
    #    -- 'range': random integers in a certain range, params = [min_int, max_int]
    #
    def random(self, dset_name, numpoints, gentype, params):
        output = []
        if gentype == 'volatility':
            output = rand_generators.rand_vol(numpoints, params[0], params[1])

        if gentype == 'range':
            output = rand_generators.rand_range(numpoints, params[0], params[1])

        hdf5 = HDF5(self.hdf5, 'a')
        hdf5.add_dset(output, dset_name, group_path=TargetData.GROUP)
        hdf5.end_access()
        return np.array(output)

    #
    # Returns target data from database 
    #
    def load(self, dset_name):
        hdf5 = HDF5(self.hdf5, 'a')
        output = hdf5.get_dset(dset_name, group_path=TargetData.GROUP)
        hdf5.end_access()
        return output


class TrainingData(InputBuilder):

    #
    # Name of group where TrainingData is stored in HDF5 database
    #
    GROUP = '/TrainingData'

    #
    # Save set of training data to database 
    #
    def save(self, dset, set_name):
        hdf5 = HDF5(self.hdf5, 'a')
        for i, x in enumerate(dset):
            hdf5.add_dset(x, str(i), group_path=TrainingData.GROUP + '/' + set_name)
        hdf5.end_access()

    #
    # Returns set of training data from database 
    #
    def load(self, set_name):
        hdf5 = HDF5(self.hdf5, 'a')
        output = hdf5.get_group(set_name, group_path=TrainingData.GROUP)
        hdf5.end_access()
        return output


class Results(InputBuilder):

    GROUP = '/Results'

    def save_matching_scores(self, results, results_name, training_data_name, target_data_name):
        hdf5 = HDF5(self.hdf5, 'a')
        hdf5.add_dset(results, results_name, group_path=Results.GROUP)
        hdf5.add_attr(results_name, 'target_data', target_data_name, group_path=Results.GROUP)
        hdf5.add_attr(results_name, 'training_data', training_data_name, group_path=Results.GROUP)
        hdf5.end_access()

    def load_matching_scores(self, results_name):
        hdf5 = HDF5(self.hdf5, 'a')
        results = hdf5.get_dset(results_name, group_path=Results.GROUP)
        target_data_name = hdf5.get_attr(results_name, 'target_data', group_path=Results.GROUP)
        training_data_name = hdf5.get_attr(results_name, 'training_data', group_path=Results.GROUP)
        hdf5.end_access()
        return results, training_data_name, target_data_name
