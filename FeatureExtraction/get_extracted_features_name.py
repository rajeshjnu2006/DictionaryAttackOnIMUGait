import os
import shutil
import warnings
import numpy as np
import pandas as pd
from GlobalParameters import DefaultParam
from Preprocess import Utilities
from python_speech_features import sigproc
from FeatureExtraction import ITheoryDomain, WaveletDomain, FrequencyDomain, TimeDomain
warnings.filterwarnings("ignore")
# Overriding the paths
DefaultParam.DATA_PATH = r'D:\GaitAuthentication30Nov2019\Storage\GenuineUsersRawData'
VALID_USER_LIST = ['User1'] # one user is sufficient to get the feature names
for USER in VALID_USER_LIST:  # For all user
    USER_PATH_IN = os.path.join(DefaultParam.DATA_PATH, USER)
    DefaultParam.MODES = ['Training']
    for MODE in DefaultParam.MODES:  # For training and testing
        MODE_PATH_IN = os.path.join(DefaultParam.DATA_PATH, USER, MODE)
        # Create the feature director if it does not exists
        for SENSOR in DefaultParam.SENSOR_FILE_LIST:  # for all sensors
            CURR_SENSOR_PATH_IN = os.path.join(MODE_PATH_IN, SENSOR)
            # print('CURR_SENSOR_PATH',CURR_SENSOR_PATH_IN)
            RAW_DATA = pd.read_csv(CURR_SENSOR_PATH_IN, sep=',', header=None, names=DefaultParam.COLUMN_NAMES)

            # The old data had different sampling rates.. here computing the sampling rate dynamically
            if USER in DefaultParam.OLD_USER_LIST:
                samplerate = Utilities.get_sampling_rate(RAW_DATA)
            else:
                samplerate = DefaultParam.NEW_SAMPLING_RATE
            print('sampling rate for ' + CURR_SENSOR_PATH_IN + ': ' + str(samplerate))
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
            # one_frame_flag = True
            for frameX, frameY, frameZ in zip(framesX, framesY, framesZ):
                # if not one_frame_flag:
                #     break
                frameM = np.sqrt(np.multiply(frameX, frameX) + np.multiply(frameY, frameY) + np.multiply(frameZ, frameZ))

                fdnamesX, fdfvectorX = FrequencyDomain.getall_fd_features(frameX, DefaultParam.NUM_BINS_FOR_FFT, samplerate, DefaultParam.WINLENGTH, DefaultParam.WINSTEP, FFTLENGTH)
                fdnamesY, fdfvectorY = FrequencyDomain.getall_fd_features(frameY, DefaultParam.NUM_BINS_FOR_FFT, samplerate, DefaultParam.WINLENGTH, DefaultParam.WINSTEP, FFTLENGTH)
                fdnamesZ, fdfvectorZ = FrequencyDomain.getall_fd_features(frameZ, DefaultParam.NUM_BINS_FOR_FFT, samplerate, DefaultParam.WINLENGTH, DefaultParam.WINSTEP, FFTLENGTH)
                fdnamesM, fdfvectorM = FrequencyDomain.getall_fd_features(frameM, DefaultParam.NUM_BINS_FOR_FFT, samplerate, DefaultParam.WINLENGTH, DefaultParam.WINSTEP, FFTLENGTH)

                itdnamesX, itdfvectorX = ITheoryDomain.getall_itd_features(frameX)
                itdnamesY, itdfvectorY = ITheoryDomain.getall_itd_features(frameY)
                itdnamesZ, itdfvectorZ = ITheoryDomain.getall_itd_features(frameZ)
                itdnamesM, itdfvectorM = ITheoryDomain.getall_itd_features(frameM)

                tdnamesX, tdfvectorX = TimeDomain.getall_td_feature(frameX)
                tdnamesY, tdfvectorY = TimeDomain.getall_td_feature(frameY)
                tdnamesZ, tdfvectorZ = TimeDomain.getall_td_feature(frameZ)
                tdnamesM, tdfvectorM = TimeDomain.getall_td_feature(frameM)

                wdfnamesX, wdfvectorX = WaveletDomain.wavelet_features(frameX, DefaultParam.MOTHER_WAVELET, DefaultParam.WAVEDEC_LEVEL)
                wdfnamesY, wdfvectorY = WaveletDomain.wavelet_features(frameY, DefaultParam.MOTHER_WAVELET, DefaultParam.WAVEDEC_LEVEL)
                wdfnamesZ, wdfvectorZ = WaveletDomain.wavelet_features(frameZ, DefaultParam.MOTHER_WAVELET, DefaultParam.WAVEDEC_LEVEL)
                wdfnamesM, wdfvectorM = WaveletDomain.wavelet_features(frameM, DefaultParam.MOTHER_WAVELET, DefaultParam.WAVEDEC_LEVEL)

                finalfdfvs = fdfvectorX + fdfvectorY + fdfvectorZ + fdfvectorM
                finalitdfvs = itdfvectorX + itdfvectorY + itdfvectorZ + itdfvectorM
                finaltdfvs = tdfvectorX + tdfvectorY + tdfvectorZ + tdfvectorM
                finalwfdfvs = wdfvectorX + wdfvectorY + wdfvectorZ + wdfvectorM

                fmatrix_freq.append(finalfdfvs)
                fmatrix_itheory.append(finalitdfvs)
                fmatrix_time.append(finaltdfvs)
                fmatrix_wavelet.append(finalwfdfvs)

            fdnames = fdnamesX+fdnamesY+fdnamesZ+fdnamesM
            itdnames = itdnamesX+itdnamesY+itdnamesZ+itdnamesM
            tdnames = tdnamesX+tdnamesY+tdnamesZ+tdnamesM
            wdfnames = wdfnamesX+wdfnamesY+wdfnamesZ+wdfnamesM
            feature_names = fdnames+itdnames+tdnames+wdfnames

            fmatrix_freq = pd.DataFrame(fmatrix_freq, columns = fdnames)
            fmatrix_itheory = pd.DataFrame(fmatrix_itheory, columns = itdnames)
            fmatrix_time = pd.DataFrame(fmatrix_time, columns = tdnames)
            fmatrix_wavelet = pd.DataFrame(fmatrix_wavelet, columns = wdfnames)
            feature_names = fdnames + itdnames + tdnames + wdfnames

            # fmatrix_freq.to_csv('fmatrix_freq'+'.csv')
            # fmatrix_itheory.to_csv('fmatrix_itheory'+'.csv')
            # fmatrix_time.to_csv('fmatrix_time'+'.csv')
            # fmatrix_wavelet.to_csv('fmatrix_wavelet'+'.csv')

        print(f'number of total features{len(feature_names)}')
        print(f'extracted features:{feature_names}')