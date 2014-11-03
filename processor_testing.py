import numpy as np
import h5py
import rand_generators as rd
import data_manipulation as dm
import timeo
from scipy.interpolate import interp1d
import features_identification as fi


START_TIMER = time.time()

################### io_functions ###############
def append_dset(descr, dset, data):
    """
    Usage:
    append_dset(file desciptor, "/GroupName/DatasetName", data)"
    GroupName is optional.
    Works with multideminisional arrays.
    Data should be numpy array, example:
    tmp = np.array([[1, 2, 3, 4, 5, 6]])
    print tmp, tmp.shape
    >>> [[1 2 3 4 5 6]] (1, 6)
    """

    data_shape = data.shape
    if (dset in descr) is True:
        dset_shape = descr[dset].shape
        new_size = data_shape[0] + dset_shape[0]

        descr[dset].resize((new_size, dset_shape[1]))
        descr[dset][dset_shape[0]:, :] = data[:]

        return 0

    else:
        descr.create_dataset(dset, data_shape,\
                dtype='Float64', maxshape=(None, data_shape[1]))

        descr[dset][0:, :] = data[:]

        return 1


def insert_dset(file_, group, dset):

    # Should be in console, data cannot be inserted
    # if dataset was not created from console????
    if (str(group) in file_) is False:
        file_.create_group(str(group))
    ###############################################

    list_n = []

    for name in file_[group]:
        list_n.append(int(name))

    if not list_n:
        last_record = 10000
    else:
        last_record = max(list_n) + 1

    file_[group + '/' + str(last_record)] = dset

    return 0
############################################################


def cum_score_01(list_):
    """
    Calculates cumulative matching score,
       precision is not known on large set's.
    """
    output = 1 - np.sum(list_)/len(list_)

    if output < 0:
        return 0.0

    return output


def matching_02(trang_set, targt_set, args_group):
    """
    Main function for matching_02
    Slightly optimized, local limits not implemented
    """
    diff = np.subtract(trang_set, targt_set)
    diff_abs = np.absolute(diff)

    diff_prx1 = np.divide(diff_abs, args_group[2])
    diff_prx2 = np.divide(diff_abs, args_group[3])

    def calc_prx(diff, diff_prx1, diff_prx2, base, expo):
        if diff < 0:
            error_prx = 1 - diff_prx1
        else:
            error_prx = 1 - diff_prx2

        if error_prx > 0:
            return (base**((2-error_prx)**expo) - base)\
                / (base**(2**expo) - base)

        return 1 + error_prx*-1

    vfunc = np.vectorize(calc_prx)

    return vfunc(diff, diff_prx1, diff_prx2, args_group[0], args_group[1])

############################################################

# Check is shear if needs normalization

def subset_transformer(hdlrs, trang_set, targt_set, matched_group,\
                       args_group, indices, counter):

    # Transformation variables
    st_en_1 = [0, 0]
    st_en_2 = [0, 0]

    counter = 0

    ### Transformation Loopwow
    # Need to select highest matched score with
    # least transformation and iterations
    output = []
    while True:

        # Transformation funcations
        # More funcation can be added via "elif"
        # and additional variables
        counter += 1

        if st_en_1[0] <= args_group[3][0]:
            st_en_1[0] += args_group[3][2]
            st_en_1[1] -= args_group[3][2]

            #print " +- Align", st_en_1[0]

            out = shear(trang_set, st_en_1[0])
            output = (normalize2(out))

            subset_processor(hdlrs, output, targt_set, matched_group,\
                                     args_group, indices, counter)


        elif st_en_2[1] <= args_group[3][1]:
            st_en_2[0] -= args_group[3][2]
            st_en_2[1] += args_group[3][2]

            #print " -+ Align", st_en_2[0]

            out = shear(trang_set, st_en_2[0])
            output = (normalize2(out))

            subset_processor(hdlrs, output, targt_set, matched_group,\
                                     args_group, indices, counter)


        else:
            #print "--- subset_transformer end"
            break


def subset_processor(hdlrs, trang_set, targt_set, matched_group,\
                     args_group, indices, counter):

    matched_array = []
    matched_values = matching_02(trang_set, targt_set, args_group[1])
    matched_array.append(matched_array)
    matched_score = cum_score_01(np.asarray(matched_values))


    if matched_score <= args_group[2][1]:
        return

    elif matched_score > args_group[2][1] and\
         matched_score < args_group[2][0] and counter == 0:

        subset_transformer(hdlrs, trang_set, targt_set,\
                           matched_group, args_group, indices, counter)
        return

    elif matched_score >= args_group[2][0] and counter == 0:

        matched_dset = np.empty([1, 5], dtype='float')
        matched_dset[0, 0] = matched_score
        matched_dset[0, 1] = indices[0][0]
        matched_dset[0, 2] = indices[0][1]-1
        matched_dset[0, 3] = indices[1]
        matched_dset[0, 4] = counter

        append_dset(HDLRS[2], "Test_V", matched_dset)
        return

    elif matched_score >= args_group[2][0] and counter >= 0:

        matched_dset = np.empty([1, 5], dtype='float')
        matched_dset[0, 0] = matched_score
        matched_dset[0, 1] = indices[0][0]
        matched_dset[0, 2] = indices[0][1]-1
        matched_dset[0, 3] = indices[1]
        matched_dset[0, 4] = counter

        append_dset(HDLRS[2], "Test_V", matched_dset)
        return

#################################################################

def shear(pattern, factor):
    """
    Initial Parameters are the fallowing:
    TRNSF_PAR = (0.0015, 0.0015, 0.00075)
    """
    output = []
    patt = []

    for i, x in enumerate(pattern):
        patt.append([float(i)] + [pattern[i]])

    line = len(pattern) / 2

    for j, x in enumerate(patt):
        patt[j][0] -= line

    tmatrix = [[1, factor], [0, 1]]
    product = np.dot(patt, tmatrix)

    for k, x in enumerate(product):
        product[k][0] += line

    for l, x in enumerate(product):
        output.append(product[l][1])

    return np.asarray(output)


def interpolate(x_array, y_array, x_len):
    temp2 = interp1d(x_array, y_array)
    output = normalize2(temp2(np.arange(x_len)))
    return np.asarray(output)


def normalize2(dset):
    """
    Normalize y data
    Input must be floats only.
    """
    _mn = dset.min()
    _mx = dset.max()
    _my = _mx - _mn

    diff = np.subtract(dset, _mn)
    norm_dset = np.divide(diff, _my)

    return norm_dset
##################################################################

# Index list is a length of Data_Set array
def indexing_generator(data_len, range_min, range_max, index_step):
    """
    Generates target data index's matrix
    """
    start = data_len[:data_len[-1]-range_max+1:index_step]

    output = []

    for start_n, min_n, max_n in\
        zip(start, np.add(start, range_min), np.add(start, range_max)):

        index_start = start_n
        index_min = min_n + index_step
        index_max = max_n + index_step

        temp_indices = np.array(data_len[index_min:index_max:index_step])
        end_indices = np.insert(temp_indices, 0, index_min-index_step, axis=0)

        for index_end in end_indices:
            output.append([index_start, index_end])

    return np.asarray(output)


def subsets_generator(hdlrs, training_data, target_data,\
                      matched_group, args_group):


    target_len = np.arange(len(hdlrs[1][target_data])+1)
    indices = indexing_generator(target_len, args_group[0][0],\
                                  args_group[0][1], args_group[0][2])

    counter1 = [0.00, 0.00]
    cntr1 = len(indices)

    for indx in indices:
        counter1[0] += 1.00

        for dset_n in hdlrs[0][training_data]:
            counter1[1] += 1

            y_set = hdlrs[0][training_data + str(dset_n)][1][:]
            x_set = hdlrs[0][training_data + str(dset_n)][0][:]
            x_len = hdlrs[0][training_data + str(dset_n)][0][-1]

            trang_set = interpolate(x_set, y_set, x_len)
            targt_set = normalize2(hdlrs[1][target_data][indx[0]:indx[1]-1])

            indices = (indx, dset_n)

            subset_processor(hdlrs, trang_set, targt_set, matched_group,\
                             args_group, indices, 0)


        print
        print "||| - New Traget Subset --///>", indx[0], indx[1]-1
        print "||| --- Elapsed Time -----///>",\
            round((time.time() - START_TIMER), ndigits=2)
        print "||| -- Processed Data ----///>",\
            str(round(counter1[0] / cntr1, ndigits=4)*100) + "%"
        print "||| --- Total Cycles -----///>", counter1[1]
        print "____________________________________"



#################################
#           ARGUMENTS           #
# From config or set in console #
#################################
# Creat Default TEST database !!!
# Database files, ONLY can be created from console
# Naming "DataType_Number'.hdf5"

FTR = h5py.File('TrainingData_1001.hdf5', "a")
FGT = h5py.File('TargetData_1001.hdf5', "a")
FMT = h5py.File('MatchedData_1001.hdf5', "a")

HDLRS = (FTR, FGT, FMT)

# Groups, Subgroups and Data sets
# _N is created when setting up GROUP's via console
#                - Only accpeted value is incremented integer
#                - Global Attributes

TRAINING_DSETS = 'TrainingDataSets_4/'
# Subgroup is created if input_builder flag is set to "new_subgroup"
#               - Auto increamented integer starting from 100001
#               - Loaded arguments are stored in subgroup attributes
#               - Attributes

TARGET_DSET = 'TargetDataSets_4/10000'
# Dataset is created every time target data is generated
#               - Auto increamented integer name starting from 100001
#               - Data originations argruments are stored in attributes
#               - Attributes

MATCHED_GROUP = 'MatchedDataGroups_1/'
# Subgroup is auto created on main process startup
#               - Auto increamented integer name starting from 100001
#               - Loaded arguments are stored as subgroup attributes
#               - Attributes

# Index [3]
TRANSFORMATION_ARGUMENTS = (1, 1, 0.00075)
# To desibale tranformation set -1

# Index [2]
ACCEPT_REJECT_THRESHOLD = (0.70, 0.55)
# [0] Accept, [1] Reject

# Index [1]
MATCHING_ARGUMENTS = (3, 2, 0.2, 0.2)
#Base, Exeponent, Top Limit, Bottom Limit

# Index [0]
SUBSETS_ARGUMENTS = (390, 390, 780)
#SUBSETS_ARGUMENTS = (60, 60, 60)
# To disable range, set min/max arguments to traget data length

ARGUMENTS_GROUP = [SUBSETS_ARGUMENTS, MATCHING_ARGUMENTS,\
                   ACCEPT_REJECT_THRESHOLD, TRANSFORMATION_ARGUMENTS]


subsets_generator(HDLRS, TRAINING_DSETS, TARGET_DSET,\
                      MATCHED_GROUP, ARGUMENTS_GROUP)


### Generate Test Data ##################################################

# for k in xrange(1000):
#     a = fi.FeatureIdentifier(rd.rand_range(390, -100, 100))
#     b = a.simplify(0.2, algo='zigzag')
#     insert_dset(FTR, TRAINING_DSETS, b)


# insert_dset(HDLRS[1], TARGET_DSET, rd.rand_range(12000, -100, 100))

#########################################################################

print "Elapsed Time", (time.time() - START_TIMER)

FTR.close()
FGT.close()
FMT.close()
