import os
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from GlobalParameters import DefaultParam
import os


def get_imp_data(filepath, feat_folder, sfile_suffix, candidate_user, valid_user_list, num_samples_imp):
    # select specified number of samples of imposter from every other user than the candidate
    possible_imp_set = set(valid_user_list) - set(candidate_user)
    count_imp = -1

    # We could pick the imp samples from other users
    # From any index, however, the first few feature vectors may not be great choice
    pick_start = 3

    for imp in possible_imp_set:
        tr_file = os.path.join(filepath, imp, 'Training', feat_folder, 'feat_clean_' + sfile_suffix + '.csv')
        ts_file = os.path.join(filepath, imp, 'Testing', feat_folder, 'feat_clean_' + sfile_suffix + '.csv')

        tr_data = pd.read_csv(tr_file, delimiter=',', index_col=0)
        ts_data = pd.read_csv(ts_file, delimiter=',', index_col=0)

        # # print('tr_file',tr_file)
        # # slicing the imp data to avoid the class imbalance
        # random.seed(DefaultParam.RANDOM_SEED)  # setting the seed so it picks the same fvectors always
        # min_num_samples = tr_data.shape[0] if tr_data.shape[0] < ts_data.shape[0] else ts_data.shape[0]
        # # The following adjustament to avoid the boundary condition issues
        # # This may result in in inconsistent number of total fvectors in the fmatrix-- but that does not matter
        # final_num_imp = (min_num_samples - pick_start) if (min_num_samples - pick_start) < num_samples_imp else num_samples_imp
        # imp_ind_list = random.sample(range(pick_start, min_num_samples), final_num_imp)

        # tr_data = tr_data.iloc[imp_ind_list, :]
        # ts_data = ts_data.iloc[imp_ind_list, :]

        tr_data = tr_data.iloc[pick_start:pick_start + num_samples_imp, :]
        ts_data = ts_data.iloc[pick_start:pick_start + num_samples_imp, :]

        count_imp = count_imp + 1  # I had forgotten this
        if count_imp == 0:
            imp_data_tr = tr_data
            imp_data_ts = ts_data
        else:
            imp_data_tr = imp_data_tr.append(tr_data, ignore_index=True)
            imp_data_ts = imp_data_ts.append(ts_data, ignore_index=True)

        if imp_data_tr.isnull().any().any() or imp_data_ts.isnull().any().any():
            raise ValueError('Found nan in current imp', imp)

    return imp_data_tr, imp_data_ts


# Generating label column for the given data matrix
def get_labels(data_matrix, label):
    if data_matrix.shape[0] > 1:
        label_column = np.empty(data_matrix.shape[0])
        label_column.fill(label)
    else:
        print('Warning! user data contains only one sample')
    return label_column


def get_normalized_data_using_minmax(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data):
    ############## Normalization or scaling ##################

    scaler = MinMaxScaler()
    # Fit only on FULL (gen and imp) training data
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    scaler.fit(training_data)

    gen_tr_data = scaler.transform(gen_tr_data)
    gen_ts_data = scaler.transform(gen_ts_data)
    # apply same transformation to test.csv data

    imp_tr_data = scaler.transform(imp_tr_data)
    imp_ts_data = scaler.transform(imp_ts_data)
    # returning as numpy arrays
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


def get_normalized_data_using_standard_scaler(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data):
    ############## Normalization or scaling ##################
    scaler = StandardScaler()
    # Fit only on FULL (gen and imp) training data
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    scaler.fit(training_data)

    gen_tr_data = scaler.transform(gen_tr_data)
    gen_ts_data = scaler.transform(gen_ts_data)
    # apply same transformation to test.csv data

    imp_tr_data = scaler.transform(imp_tr_data)
    imp_ts_data = scaler.transform(imp_ts_data)

    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


def get_balanced_data(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data):
    ### IT DOES NOT MAKE SENSE TO RESAMPLE THE TESTING DATA === AND YOU MUST NOT BECAUSE THE INFORMATION YOU WILL USE
    ### WONT BE AVAILABLE AT THE TESTING TIME ie. IMPSAMPLES...! EVEN IF YOU HAVE TO DO IT YOU SHOULD DO IT WITH TRAINING IMP SAMPLES
    ### JUST RESAMPLE THE TRAINING SO THE CLASSIFICATION MODEL IS NOT BIASED

    train_X = np.vstack((gen_tr_data, imp_tr_data))
    train_Y = np.concatenate((get_labels(gen_tr_data, 1), get_labels(imp_tr_data, -1)))
    DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING = min(DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING, gen_tr_data.shape[0] - 1)  # number of feat maybe less than nn in rarest cases

    # print(f'Using {DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING} NN for SMOTE')
    oversampling = SMOTE(sampling_strategy=1.0, random_state=DefaultParam.RANDOM_SEED,
                         k_neighbors=DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING)
    # The fit_resample function resamples the given data and makes the classes balanced same as fit_sample --backward compatibility
    # print(f"BEFORE SMOTE, gen_tr_data:{gen_tr_data.shape}, imp_tr_data:{imp_tr_data.shape}")
    # print(f"BEFORE SMOTE, gen_ts_data:{gen_ts_data.shape}, imp_ts_data:{imp_ts_data.shape}")

    train_X_resampled, train_Y_resampled = oversampling.fit_resample(train_X, train_Y)

    # Separating genuine and impostor feature vectors, this can be done in numpy format
    training_data = np.column_stack((train_X_resampled, train_Y_resampled))

    gen_tr_data_new = training_data[training_data[:, -1] == 1]
    imp_tr_data_new = training_data[training_data[:, -1] == -1]

    gen_tr_data = gen_tr_data_new[:, :-1]
    imp_tr_data = imp_tr_data_new[:, :-1]
    # print(f"AFTER SMOTE, gen_tr_data:{gen_tr_data.shape}, imp_tr_data:{imp_tr_data.shape}")
    # print(f"AFTER SMOTE, gen_ts_data:{gen_ts_data.shape}, imp_ts_data:{imp_ts_data.shape}")

    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


def get_data(DATA_PATH, FEATURE_FOLDER, SENSOR_PREFIX, VALID_USER_LIST, USER, NUM_SAMPLE_IMP):
    candidate_user = USER
    tr_file = os.path.join(DATA_PATH, candidate_user, 'Training', FEATURE_FOLDER,
                           'feat_clean_' + SENSOR_PREFIX + '.csv')
    ts_file = os.path.join(DATA_PATH, candidate_user, 'Testing', FEATURE_FOLDER, 'feat_clean_' + SENSOR_PREFIX + '.csv')

    # Preparing genuine train and test.csv data
    gen_tr_data = pd.read_csv(tr_file, delimiter=',', index_col=0)
    gen_ts_data = pd.read_csv(ts_file, delimiter=',', index_col=0)

    # Getting impostor data
    imp_tr_data, imp_ts_data = get_imp_data(DATA_PATH, FEATURE_FOLDER, SENSOR_PREFIX, USER, VALID_USER_LIST,
                                            NUM_SAMPLE_IMP)

    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


def get_data_for_attack_test(DATA_PATH, ATTACK_ATTEMPTS, FEATURE_FOLDER, SENSOR_PREFIX, VALID_USER_LIST, USER_ID,
                             NUM_SAMPLE_IMP):
    candidate_user = 'User' + str(USER_ID)
    TR_FILE_PATH = os.path.join(DATA_PATH, candidate_user, 'Training', FEATURE_FOLDER,
                                'feat_clean_' + SENSOR_PREFIX + '.csv')
    TS_FILE_PATH = os.path.join(DATA_PATH, candidate_user, 'Testing', FEATURE_FOLDER,
                                'feat_clean_' + SENSOR_PREFIX + '.csv')

    # Preparing genuine train and test.csv data
    gen_tr_data = np.loadtxt(TR_FILE_PATH, delimiter=',')
    gen_ts_data = np.loadtxt(TS_FILE_PATH, delimiter=',')

    imp_tr_data, imp_ts_data = get_imp_data(DATA_PATH, FEATURE_FOLDER, SENSOR_PREFIX, USER_ID, VALID_USER_LIST,
                                            NUM_SAMPLE_IMP)

    ATTACK_TEST_FILE_PATH = os.path.join(DATA_PATH, candidate_user, ATTACK_ATTEMPTS[0], FEATURE_FOLDER,
                                         'feat_clean_' + SENSOR_PREFIX + '.csv')
    # print('Attack data is coming from:', ATTACK_TEST_FILE_PATH)
    com_imp_ts_data = np.loadtxt(ATTACK_TEST_FILE_PATH, delimiter=',')

    for ATTACK_ATTEMPT in ATTACK_ATTEMPTS[1:]:
        ATTACK_TEST_FILE_PATH = os.path.join(DATA_PATH, candidate_user, ATTACK_ATTEMPT, FEATURE_FOLDER,
                                             'feat_clean_' + SENSOR_PREFIX + '.csv')
        # print('Attack data is coming from:', ATTACK_TEST_FILE_PATH)
        temp = np.loadtxt(ATTACK_TEST_FILE_PATH, delimiter=',')
        com_imp_ts_data = np.vstack((com_imp_ts_data, temp))

    if np.isnan(np.min(com_imp_ts_data)):
        print('Found nan in current imp', ATTACK_TEST_FILE_PATH)
        exit()

    return gen_tr_data, gen_ts_data, imp_tr_data, com_imp_ts_data


# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=3, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


###################################################################################################
#############THIS IS BEING CREATED FOR DICTIONARY EFFORT ATTACK####################################
# Basically the following functions would modularize things a little more
def get_combined_data_ZEFFORT(user, imposter_list, curr_comb, num_imp):
    # preparing data based on what domain of features we have to include from what sensors
    for sensor in curr_comb:
        for fdomain in DefaultParam.FEATURE_DOMAIN_LIST:
            if 'com_gen_tr_data' not in locals():
                com_gen_tr_data, com_gen_ts_data, com_imp_tr_data, com_imp_ts_data = get_data(
                    DefaultParam.FEATUREFILE_PATH, fdomain,
                    sensor, imposter_list, user, num_imp)
            else:
                gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data = get_data(DefaultParam.FEATUREFILE_PATH, fdomain,
                                                                              sensor, imposter_list, user, num_imp)
                min_gen_tr_samples = min(com_gen_tr_data.shape[0], gen_tr_data.shape[0])
                min_gen_ts_samples = min(com_gen_ts_data.shape[0], gen_ts_data.shape[0])
                min_imp_tr_samples = min(com_imp_tr_data.shape[0], imp_tr_data.shape[0])
                min_imp_ts_samples = min(com_imp_ts_data.shape[0], imp_ts_data.shape[0])
                com_gen_tr_data = pd.concat(
                    [com_gen_tr_data.iloc[0:min_gen_tr_samples, :], gen_tr_data.iloc[0:min_gen_tr_samples, :]], axis=1)
                com_gen_ts_data = pd.concat(
                    [com_gen_ts_data.iloc[0:min_gen_ts_samples, :], gen_ts_data.iloc[0:min_gen_ts_samples, :]], axis=1)
                com_imp_tr_data = pd.concat(
                    [com_imp_tr_data.iloc[0:min_imp_tr_samples, :], imp_tr_data.iloc[0:min_imp_tr_samples, :]], axis=1)
                com_imp_ts_data = pd.concat(
                    [com_imp_ts_data.iloc[0:min_imp_ts_samples, :], imp_ts_data.iloc[0:min_imp_ts_samples, :]], axis=1)
    return com_gen_tr_data, com_gen_ts_data, com_imp_tr_data, com_imp_ts_data


def get_combined_imp_data_DICTATTACK(user, curr_comb):
    # preparing data based on what domain of features we have to include from what sensors
    for sensor in curr_comb:
        for fdomain in DefaultParam.FEATURE_DOMAIN_LIST:
            if 'com_data' not in locals():
                com_data = pd.read_csv(
                    os.path.join(DefaultParam.DICT_FEATUREFILE_PATH, user, fdomain, 'feat_clean_' + sensor + '.csv'),
                    delimiter=',', index_col=0)
            else:
                temp = pd.read_csv(
                    os.path.join(DefaultParam.DICT_FEATUREFILE_PATH, user, fdomain, 'feat_clean_' + sensor + '.csv'),
                    delimiter=',', index_col=0)
                min_samples = min(com_data.shape[0], temp.shape[0])
                com_data = pd.concat([com_data.iloc[0:min_samples, :], temp.iloc[0:min_samples, :]], axis=1)

    return com_data


def get_normalized_data_using_standard_scaler_DICTATTACK(gen_tr_data, imp_tr_data, attack_data):
    ############## Normalization or scaling ##################
    scaler = StandardScaler()
    # Fit only on FULL (gen and imp) training data
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    scaler.fit(training_data)
    attack_data = scaler.transform(attack_data)
    return attack_data


def get_combined_imp_data_HIGHATTACK(user, attempt, curr_comb):
    # preparing data based on what domain of features we have to include from what sensors
    for sensor in curr_comb:
        for fdomain in DefaultParam.FEATURE_DOMAIN_LIST:
            if 'com_data' not in locals():
                com_data = pd.read_csv(
                    os.path.join(DefaultParam.HIGH_FEATUREFILE_PATH, user, attempt, fdomain,
                                 'feat_clean_' + sensor + '.csv'),
                    delimiter=',', index_col=0)
            else:
                temp = pd.read_csv(
                    os.path.join(DefaultParam.HIGH_FEATUREFILE_PATH, user, attempt, fdomain,
                                 'feat_clean_' + sensor + '.csv'),
                    delimiter=',', index_col=0)
                min_samples = min(com_data.shape[0], temp.shape[0])
                com_data = pd.concat([com_data.iloc[0:min_samples, :], temp.iloc[0:min_samples, :]], axis=1)

    return com_data


def get_normalized_data_using_standard_scaler_HIGHATTACK(gen_tr_data, imp_tr_data, attack_data):
    ############## Normalization or scaling ##################
    scaler = StandardScaler()
    # Fit only on FULL (gen and imp) training data
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    scaler.fit(training_data)
    attack_data = scaler.transform(attack_data)
    return attack_data
