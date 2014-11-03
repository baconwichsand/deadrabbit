import numpy as np
import random
import os
import time
import pdb

#################################
##### DataManipulation.py
#################################
import DataManipulation
reload(DataManipulation)

checksum = []

function = 'remove_x'
argument = np.array([[0., 1.], [1., 3.], [2., 1.]])
ideal_output = np.array([1., 3., 1.])
real_output = DataManipulation.remove_x(argument)
a = (ideal_output == real_output).all()
checksum.append(a)

print function, a

function = 'add_x'
argument = np.array([1., 3., 1.])
ideal_output = np.array([[0, 1.], [1, 3.], [2, 1.]])
real_output = DataManipulation.add_x(argument)
a = (ideal_output == real_output).all()
checksum.append(a)

print function, a

function = 'normalize'
argument = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
ideal_output = np.array([0., 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.])
real_output = np.around(DataManipulation.normalize(argument), 4)
a = (ideal_output == real_output).all()
checksum.append(a)

print function, a

function = 'interpolate'
argument = np.array([[1., 10.], [5., 14.], [8., 4.], [10., 19.], [12., 8.]])
ideal_output = np.array([10., 11., 12., 13., 14., 10.6667, 7.3333, 4., 11.5, 19., 13.5, 8.])
real_output = np.around(DataManipulation.interpolate(argument), 4)
a = (ideal_output == real_output).all()
checksum.append(a)

print function, a

function = 'eq_scale'
argument1 = np.random.random(57)
DataManipulation.add_x(argument1)
argument2 = np.random.random(22)
ideal_output = [1254, 1254, 1, 1]
real_output = [len(DataManipulation.eq_scale(argument1, argument2)[0]), len(DataManipulation.eq_scale(argument1, argument2)[1]), DataManipulation.eq_scale(argument1, argument2)[0].ndim, DataManipulation.eq_scale(argument1, argument2)[1].ndim]
a = (ideal_output == real_output)
checksum.append(a)

print function, a

total = np.array(checksum)
print "====================================="
print "DataManipulation.py", bool(total.all())
print "====================================="


#################################
##### IOOperations.py
#################################
import IOOperations
reload(IOOperations)

############## HDF5
import h5py

checksum1 = []

function = '__init__'
arguments = ('testing.hdf5', 'a')
hdf5_test = IOOperations.HDF5(*arguments)
ideal_output = '<class \'h5py._hl.files.File\'>'
real_output = str(type(hdf5_test.operator))
hdf5_test.operator.close()
a = (ideal_output == real_output)
checksum1.append(a)

print function, a

function = 'end_access'
arguments = None
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
ideal_output = False
hdf5_test.end_access()
val = False
if hdf5_test.operator:
    val = True
real_output = val
a = (ideal_output == real_output)
checksum1.append(a)

print function, a


function = 'add_dset'
name = str(random.randint(1000000, 500000000))
arguments = (np.array([1, 2, 3, 4]), name)
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
hdf5_test.add_dset(*arguments, group_path='test01')
hdf5_test.end_access()
ideal_output = np.array([1, 2, 3, 4])
r = h5py.File('testing.hdf5', 'a')
real_output = r['/test01/' + name][...]
r.close()
a = (ideal_output == real_output).all()
checksum1.append(a)

print function, a


function = 'get_dset'
r = h5py.File('testing.hdf5', 'a')
name = str(random.randint(1000000, 50000000))
r['/test02/' + name] = np.array([1, 2, 3, 4])
r.close()
arguments = name
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
ideal_output = np.array([1, 2, 3, 4])
real_output = hdf5_test.get_dset(arguments, group_path='/test02')
hdf5_test.end_access()
a = (ideal_output == real_output).all()
checksum1.append(a)

print function, a


function = 'get_group'
r = h5py.File('testing.hdf5', 'a')
name1 = str(random.randint(1000000, 50000000))
name2 = str(random.randint(1000000, 50000000))
r['/test03/' + name1] = np.array([1, 2, 3, 4])
r['/test03/' + name2] = np.array([5, 6, 7, 8])
r.close()
arguments = 'test03'
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
ideal_output = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
real_output = hdf5_test.get_group(arguments)
hdf5_test.end_access()
val = True
for i, x in enumerate(ideal_output):
    if not (ideal_output[i] == real_output[i]).all():
        val = False
a = val
checksum1.append(a)

print function, a


function = 'new_session'
arguments = None
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
hdf5_test.new_session()
hdf5_test.end_access()
ideal_output = '/' + time.strftime('%c')
r = h5py.File('testing.hdf5', 'a')
real_output = str(r['/' + time.strftime('%c')].name)
r.close()
a = (ideal_output == real_output)
checksum1.append(a)

print function, a



total = np.array(checksum1)
print "------------->HDF5", bool(total.all())


checksum2 = []

############## Config
import ConfigParser

function = 'get'
arguments = ('test01', 'test01')
ideal_output = 3
cp_test = IOOperations.Config('testing.cfg')
real_output = cp_test.get(*arguments, dtype=int)
a = (ideal_output == real_output)
checksum2.append(a)

print function, a

function = 'get_section'
arguments = ('test01')
ideal_output = ['3', '0.04', 'blah']
cp_test = IOOperations.Config('testing.cfg')
real_output = cp_test.get_section(arguments)
a = (ideal_output == real_output)
checksum2.append(a)

print function, a


total = np.array(checksum2)
print "------------->Config", bool(total.all())


############## CSV

checksum3 = []

import csv

function = 'read_csv_rows'
arguments = 0
ideal_output = [[0., 10.], [1., 12.], [2., 8.], [3., 6.], [4., 14.], [5., 16.]]
test_csv = IOOperations.CSV('testing.csv')
real_output = test_csv.read_csv_rows()
a = (ideal_output == real_output).all()
checksum3.append(a)

print function, a


total = np.array(checksum3)
print "------------->CSV", bool(total.all())


total = np.array(checksum1 + checksum2 + checksum3)
print "====================================="
print "IOOperations.py", bool(total.all())
print "====================================="



#################################
##### InputBuilder.py
#################################
import InputBuilder
reload(InputBuilder)

############## TargetData

checksum1 = []

function = 'csv'
arguments = ('testing.csv', 'csv01')
ideal_output = [10., 12., 8., 6., 14., 16.]
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
tdata_test = InputBuilder.TargetData('testing.hdf5')
tdata_test.csv(*arguments)
real_output = hdf5_test.get_dset('csv01', group_path='/TargetData')
hdf5_test.end_access()
a = (ideal_output == real_output).all()
checksum1.append(a)

print function, a


function = 'random'
arguments = ('rand01', 100, 'volatility', [1000, 0.02])
ideal_output = (100, type(np.array([0.01, 0.02])[0]))
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
tdata_test = InputBuilder.TargetData('testing.hdf5')
tdata_test.random(*arguments)
output = hdf5_test.get_dset('rand01', group_path='/TargetData')
real_output = (len(output), type(output[0]))
hdf5_test.end_access()
a = (ideal_output == real_output)
checksum1.append(a)

print function, a


function = 'load'
arguments = 'csv01'
ideal_output = [10., 12., 8., 6., 14., 16.]
tdata_test = InputBuilder.TargetData('testing.hdf5')
real_output = tdata_test.load(arguments)
a = (ideal_output == real_output).all()
checksum1.append(a)
print function, a



total = np.array(checksum1)
print "------------->TargetData", bool(total.all())



############## TrainingData

checksum2 = []


function = 'save'
arguments = ([[[0, 1], [1, 2], [2, 3]], [[3, 4], [4, 5], [5, 6]], [[6, 7], [7, 8], [8, 9]]], 'test01')
ideal_output = [[[0, 1], [1, 2], [2, 3]], [[3, 4], [4, 5], [5, 6]], [[6, 7], [7, 8], [8, 9]]]
hdf5_test = IOOperations.HDF5('testing.hdf5', 'a')
tdata_test = InputBuilder.TrainingData('testing.hdf5')
tdata_test.save(*arguments)
real_output = tdata_test.load(arguments)
a = (ideal_output == real_output).all()
checksum1.append(a)

print function, a

function = 'load'
arguments = 'csv01'
ideal_output = [10., 12., 8., 6., 14., 16.]
tdata_test = InputBuilder.TargetData('testing.hdf5')
real_output = tdata_test.load(arguments)
a = (ideal_output == real_output).all()
checksum1.append(a)

print function, a




total = np.array(checksum2)
print "------------->TrainingData", bool(total.all())


total = np.array(checksum1 + checksum2)
print "====================================="
print "InputBuilder.py", bool(total.all())
print "====================================="



#################################
##### FeatIdent.py
#################################
import FeatIdent
reload(FeatIdent)

checksum = []

function = 'simplify'
arguments = ''
ideal_output = True
tdata_test = False
real_output = None
a = (ideal_output == real_output)
checksum.append(a)

print function, a


function = 'fractal'
arguments = ''
ideal_output = True
tdata_test = False
real_output = None
a = (ideal_output == real_output)
checksum.append(a)

print function, a


function = 'zigzag'
arguments = ''
ideal_output = True
tdata_test = False
real_output = None
a = (ideal_output == real_output)
checksum.append(a)

print function, a



total = np.array(checksum)
print "====================================="
print "FeatIdent.py", bool(total.all())
print "====================================="



#################################
##### matching.py
#################################
import Matching
reload(Matching)

checksum = []

function = 'error_match'
arguments = ''
ideal_output = True
tdata_test = False
real_output = None
a = (ideal_output == real_output)
checksum.append(a)

print function, a


function = 'LCS'
arguments = ''
ideal_output = True
tdata_test = False
real_output = None
a = (ideal_output == real_output)
checksum.append(a)

print function, a


total = np.array(checksum)
print "====================================="
print "Matching.py", bool(total.all())
print "====================================="



#################################
##### Visualization.py
#################################
import Visualization
reload(Visualization)

checksum = []

function = '__init__'
arguments = ''
ideal_output = True
tdata_test = False
real_output = None
a = (ideal_output == real_output)
checksum.append(a)

print function, a


total = np.array(checksum)
print "====================================="
print "Visualization.py", bool(total.all())
print "====================================="



os.remove('testing.hdf5')
