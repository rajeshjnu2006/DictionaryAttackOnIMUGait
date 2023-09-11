import os
import shutil
from matplotlib import pyplot as plt
import math
import warnings
import numpy as np
import pandas as pd
from GlobalParameters import DefaultParam
from Preprocess import Utilities
from python_speech_features import sigproc
from FeatureExtraction import ITheoryDomain, WaveletDomain, FrequencyDomain, TimeDomain
warnings.filterwarnings("ignore")
# Overriding the paths
DefaultParam.DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'Storage', 'RawDataDictionary')
DefaultParam.FEATURE_PATH = os.path.join(os.path.dirname(os.getcwd()), 'Storage', 'FeatureFilesDictionary')

USER_LIST = os.listdir(DefaultParam.DATA_PATH)
USER_LIST = sorted(USER_LIST)
INVALID_USER_LIST = []
VALID_USER_LIST = set(USER_LIST) - set(INVALID_USER_LIST)
# Sorting the list of users for the sake of keeping track how many finished
VALID_USER_LIST = sorted(VALID_USER_LIST)
print('VALID_USER_LIST',VALID_USER_LIST)
for USER in VALID_USER_LIST:  # For all user
    USER_PATH_IN = os.path.join(DefaultParam.DATA_PATH, USER)
    USER_PATH_OUT = os.path.join(DefaultParam.FEATURE_PATH, USER)
    if os.path.exists(USER_PATH_OUT):
        raise FileExistsError("The output directory already exists.. delete manually")
    else:
        os.mkdir(USER_PATH_OUT)
        print(f'Created a new directory!')

    PATH_IN = os.path.join(DefaultParam.DATA_PATH, USER)
    FREQ_PATH = os.path.join(DefaultParam.FEATURE_PATH, USER, 'FrequencyFeatures')
    IT_PATH = os.path.join(DefaultParam.FEATURE_PATH, USER, 'ITheoryFeatures')
    TIME_PATH = os.path.join(DefaultParam.FEATURE_PATH, USER, 'TimeFeatures')
    WAVE_PATH = os.path.join(DefaultParam.FEATURE_PATH, USER, 'WaveletFeatures')
    # Create the feature directories if they do not exist
    try:  # Create target Directory
        os.mkdir(FREQ_PATH)
        os.mkdir(IT_PATH)
        os.mkdir(TIME_PATH)
        os.mkdir(WAVE_PATH)
    except FileExistsError:
        print("The feature folders already exists")

    for SENSOR in DefaultParam.SENSOR_FILE_LIST:  # for all sensors
        CURR_SENSOR_PATH_IN = os.path.join(PATH_IN, SENSOR)
        CS_FREQ_PATH = os.path.join(FREQ_PATH, 'feat_' + SENSOR.replace(".txt", ".csv"))
        CS_IT_PATH = os.path.join(IT_PATH, 'feat_' + SENSOR.replace(".txt", ".csv"))
        CS_TIME_PATH = os.path.join(TIME_PATH, 'feat_' + SENSOR.replace(".txt", ".csv"))
        CS_WAVE_PATH = os.path.join(WAVE_PATH, 'feat_' + SENSOR.replace(".txt", ".csv"))
        print('CURR_SENSOR_PATH',CURR_SENSOR_PATH_IN, ', OUT_PATH:',CS_FREQ_PATH)
        RAW_DATA = pd.read_csv(CURR_SENSOR_PATH_IN, sep=',', header=None, names=DefaultParam.NEW_COLUMN_NAMES)
        samplerate = DefaultParam.NEW_SAMPLING_RATE
        print('sampling rate for ' + CURR_SENSOR_PATH_IN + ': ' + str(samplerate))

        ######################################################################
        signalX = RAW_DATA['x']
        signalY = RAW_DATA['y']
        signalZ = RAW_DATA['z']

        # Sliding window based cutting of the signal
        framesX = sigproc.framesig(signalX, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
        framesY = sigproc.framesig(signalY, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
        framesZ = sigproc.framesig(signalZ, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)

        if framesX.shape[1] > 64 and framesX.shape[1] < 128:
            FFTLENGTH = 128
        elif framesX.shape[1] > 128 and framesX.shape[1] < 256:
            FFTLENGTH = 256
        else:
            FFTLENGTH = 512

        # The  sigproc.framesig function fills zeroes in the last unavailable points
        # we dont want that
        framesX = framesX[:-1, :]
        framesY = framesY[:-1, :]
        framesZ = framesZ[:-1, :]

        fmatrix_freq = []
        fmatrix_itheory = []
        fmatrix_time = []
        fmatrix_wavelet = []

        for frameX, frameY, frameZ in zip(framesX, framesY, framesZ):
            ############SMOOTHING THE DATA BEFORE ANY FEATURE EXTRACTION
            ############EVEN BEFORE COMPUTATION OF THE M -- ELSE ERROR PROPOGATE
            # smoothing the signal with 5% of total obtained in every second
            percent_smooth = 0.05
            smoothing_span = math.ceil(samplerate * percent_smooth + 1)
            framesX = Utilities.moving_average(frameX, smoothing_span)
            framesY = Utilities.moving_average(frameY, smoothing_span)
            framesZ = Utilities.moving_average(frameZ, smoothing_span)

            # plt.plot(framesX)
            # plt.show()

            frameM = np.sqrt(frameX * frameX + np.multiply(frameY, frameY) + np.multiply(frameZ, frameZ))

            fdnamesX, fdfvectorX = FrequencyDomain.getall_fd_features(frameX, DefaultParam.NUM_BINS_FOR_FFT,
                                                                      samplerate, DefaultParam.WINLENGTH,
                                                                      DefaultParam.WINSTEP, FFTLENGTH)
            fdnamesY, fdfvectorY = FrequencyDomain.getall_fd_features(frameY, DefaultParam.NUM_BINS_FOR_FFT,
                                                                      samplerate, DefaultParam.WINLENGTH,
                                                                      DefaultParam.WINSTEP, FFTLENGTH)
            fdnamesZ, fdfvectorZ = FrequencyDomain.getall_fd_features(frameZ, DefaultParam.NUM_BINS_FOR_FFT,
                                                                      samplerate, DefaultParam.WINLENGTH,
                                                                      DefaultParam.WINSTEP, FFTLENGTH)
            fdnamesM, fdfvectorM = FrequencyDomain.getall_fd_features(frameM, DefaultParam.NUM_BINS_FOR_FFT,
                                                                      samplerate, DefaultParam.WINLENGTH,
                                                                      DefaultParam.WINSTEP, FFTLENGTH)

            itdnamesX, itdfvectorX = ITheoryDomain.getall_itd_features(frameX)
            itdnamesY, itdfvectorY = ITheoryDomain.getall_itd_features(frameY)
            itdnamesZ, itdfvectorZ = ITheoryDomain.getall_itd_features(frameZ)
            itdnamesM, itdfvectorM = ITheoryDomain.getall_itd_features(frameM)

            tdnamesX, tdfvectorX = TimeDomain.getall_td_feature(frameX)
            tdnamesY, tdfvectorY = TimeDomain.getall_td_feature(frameY)
            tdnamesZ, tdfvectorZ = TimeDomain.getall_td_feature(frameZ)
            tdnamesM, tdfvectorM = TimeDomain.getall_td_feature(frameM)

            wdfnamesX, wdfvectorX = WaveletDomain.wavelet_features(frameX, DefaultParam.MOTHER_WAVELET,
                                                                   DefaultParam.WAVEDEC_LEVEL)
            wdfnamesY, wdfvectorY = WaveletDomain.wavelet_features(frameY, DefaultParam.MOTHER_WAVELET,
                                                                   DefaultParam.WAVEDEC_LEVEL)
            wdfnamesZ, wdfvectorZ = WaveletDomain.wavelet_features(frameZ, DefaultParam.MOTHER_WAVELET,
                                                                   DefaultParam.WAVEDEC_LEVEL)
            wdfnamesM, wdfvectorM = WaveletDomain.wavelet_features(frameM, DefaultParam.MOTHER_WAVELET,
                                                                   DefaultParam.WAVEDEC_LEVEL)

            finalfdfvs = fdfvectorX + fdfvectorY + fdfvectorZ + fdfvectorM
            finalitdfvs = itdfvectorX + itdfvectorY + itdfvectorZ + itdfvectorM
            finaltdfvs = tdfvectorX + tdfvectorY + tdfvectorZ + tdfvectorM
            finalwfdfvs = wdfvectorX + wdfvectorY + wdfvectorZ + wdfvectorM

            fmatrix_freq.append(finalfdfvs)
            fmatrix_itheory.append(finalitdfvs)
            fmatrix_time.append(finaltdfvs)
            fmatrix_wavelet.append(finalwfdfvs)

        fdnames = fdnamesX + fdnamesY + fdnamesZ + fdnamesM
        itdnames = itdnamesX + itdnamesY + itdnamesZ + itdnamesM
        tdnames = tdnamesX + tdnamesY + tdnamesZ + tdnamesM
        wdfnames = wdfnamesX + wdfnamesY + wdfnamesZ + wdfnamesM
        feature_names = fdnames + itdnames + tdnames + wdfnames

        fmatrix_freq = pd.DataFrame(fmatrix_freq, columns=fdnames)
        fmatrix_itheory = pd.DataFrame(fmatrix_itheory, columns=itdnames)
        fmatrix_time = pd.DataFrame(fmatrix_time, columns=tdnames)
        fmatrix_wavelet = pd.DataFrame(fmatrix_wavelet, columns=wdfnames)

        fmatrix_freq.to_csv(CS_FREQ_PATH)
        fmatrix_itheory.to_csv(CS_IT_PATH)
        fmatrix_time.to_csv(CS_TIME_PATH)
        fmatrix_wavelet.to_csv(CS_WAVE_PATH)