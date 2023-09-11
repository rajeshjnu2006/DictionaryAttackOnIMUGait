import os
CURRENT_PATH = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(CURRENT_PATH,'Storage', 'RawDataGenuine')
FEATUREFILE_PATH = os.path.join(CURRENT_PATH,'Storage', 'FeatureFilesGenuine')
DICT_FEATUREFILE_PATH = os.path.join(CURRENT_PATH,'Storage', 'FeatureFilesDictionary')
HIGH_FEATUREFILE_PATH = os.path.join(CURRENT_PATH,'Storage', 'FeatureFilesHigh')

print(HIGH_FEATUREFILE_PATH)
print(os.getcwd())

# Put in the list the domains that you want to fuse
BAD_USERS = []
REMOVE_BAD_USERS = False
# FEATURE_DOMAIN_LIST = ['TimeFeatures', 'FrequencyFeatures', 'ITheoryFeatures', 'WaveletFeatures']
FEATURE_DOMAIN_LIST_SHORT = ['time', 'freq', 'info', 'wave']
FEATURE_DOMAIN_LIST = ['TimeFeatures', 'FrequencyFeatures']

# Using only two domains.. these domains are rampantly used in literature ...we can use other domains if the need be
# OR ttest if those domains offer
SENSOR_LIST = ['LAcc', 'Gyr', 'Mag', 'RVec']
NUM_SAMPLE_IMP = 5
NEW_SAMPLING_RATE = 46
RANDOM_SEED = 21
# Preprocessing
CLASS_BALANCING = True
NNEIGHBORS_FOR_OVERSAMPLING = 7  # For All results till Dec 19 , this was set to 5
NORMALIZE_FEATURES = True
# Parameters for feature extraction
APPLY_SUPERVISED_FSELECTION = True
APPLY_UNIVARIATE_FSELECTION = True
APPLY_RANDFOR_FSELECTION = True
APPLY_PCA = False  # Giving good results
APPLY_LDA = False  # Giving good results too but we wont be using this ... not a part of the proposal.. dont deviate
COMP_PERCENTILE = 1
CORR_THR_FEXTRACTION = 0.2
PCA_PER_VAR = 0.97
LDA_PER_CENT = 0.50
WINLENGTH = 8  # in seconds
WINSTEP = int(WINLENGTH / 2)  # in seconds
FFTLENGTH = 512
MOTHER_WAVELET = 'db5'
WAVEDEC_LEVEL = 5
NUM_BINS = int(WINLENGTH*2)
NUM_BINS_FOR_TD = int(WINLENGTH*2)
MODES = ['Training', 'Testing', 'Attempt1', 'Attempt2', 'Attempt3']
OLD_COLUMN_NAMES = ['x', 'y', 'z', 'unix_timestamps', 'timestamps']
NEW_COLUMN_NAMES = ['x', 'y', 'z']
# NEW_COLUMN_NAMES = ['x', 'y', 'z']
COLUMN_NAMES_PROFILING = ['x', 'y', 'z']
SENSOR_FILE_LIST = ['clean_Gyr.txt', 'clean_LAcc.txt', 'clean_Mag.txt', 'clean_RVec.txt']
NUM_BINS_FOR_FFT = int(FFTLENGTH/2)
SAMPLE_LENGTH_ITD = 1
ORDER_OF_PERMUTE_ITD = 2  # (m is the embedded dimension)
DELAY_ITD = 5
SCALE_ITD = 4
MSCALE_ITD = 4
# Attack parameters
ATTACK_ATTEMPTS = ['Attempt1', 'Attempt2', 'Attempt3']
OLD_USER_LIST = []
for i in range(1, 30, 1):
    OLD_USER_LIST.append('User' + str(i))

HIGHEFFORT_USER_LIST = [] # 18 old users have been attacked under high effort scenario
for i in range(1, 19, 1):
    HIGHEFFORT_USER_LIST.append('User' + str(i))
