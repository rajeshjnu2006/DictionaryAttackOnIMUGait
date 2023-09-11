### What changes when the GCAT settings change???
# For example, which component (x,y, and z) change when we increase the speed??
import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
# Adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from GlobalParameters import DefaultParam
from Preprocess import Prep
from python_speech_features import sigproc
from scipy import signal

warnings.filterwarnings("ignore")
imdb_path = 'F:\ThesisFinal4April2019\Storage\PreprocessedProfilingAnonymous'
imitators = os.listdir(imdb_path)
exclude_imt_list = ['Imitator4', 'Imitator6']
for item in exclude_imt_list:
    imitators.remove(item)

attack_sensor = 'clean_LAcc.txt'
samplerate = 46
num_bins = 20  # Acc varies between -10g to 10 g
# Taking about 40 sec of data points | (46*40) 1840 data points
begin_end_remove_data_points = 230  # Taking away 5 seconds of data
total_data_points = 2070
gcat_list = ['slength', 'swidth', 'tlift', 'speed']
for imt in imitators:  ## Traversing over all imitators
    curr_imp_folder = os.path.join(imdb_path, imt)
    settings = os.listdir(curr_imp_folder)
    speed_sett_list = []
    slength_sett_list = []
    swidth_sett_list = []
    tlift_sett_list = []
    for sett in settings:
        if 'speed' in sett:
            speed_sett_list.append(sett)
        if 'slength' in sett:
            slength_sett_list.append(sett)
        if 'swidth' in sett:
            swidth_sett_list.append(sett)
        if 'tlift' in sett:
            tlift_sett_list.append(sett)

    for item in gcat_list:
        if item == 'speed':
            # Using only odd indices, speed was perfectly ordered to so no manual ordering
            set_list = speed_sett_list[0::2]
        elif item == 'slength':
            set_list = ['slength_short', 'slength_normal', 'slength_long', 'slength_longer']
        elif item == 'swidth':
            set_list = ['swidth_close', 'swidth_normal', 'swidth_wide', 'swidth_wider']
        else:
            set_list = ['tlift_back', 'tlift_normal', 'tlift_front', 'tlift_up']
        #############################
        plt.figure(imt + item + 'histogram', figsize=(12, 6))
        num_sub_plots = len(set_list)
        subplot_count = 0
        print('set_list', set_list)
        for sett in set_list:
            curr_sample_location = os.path.join(curr_imp_folder, sett)
            # Computing the feature vector for the current imitator at current setting
            FILE_PATH = os.path.join(curr_sample_location, attack_sensor)
            RAW_DATA = pd.read_csv(FILE_PATH, sep=',', header=None, names=DefaultParam.COLUMN_NAMES)

            signalX = RAW_DATA['x']
            signalY = RAW_DATA['y']
            signalZ = RAW_DATA['z']

            signalX = signalX[begin_end_remove_data_points:total_data_points]
            signalY = signalY[begin_end_remove_data_points:total_data_points]
            signalZ = signalZ[begin_end_remove_data_points:total_data_points]

            signalX = Prep.smooth(signalX, window_len=int(samplerate * 10 / 100))
            signalY = Prep.smooth(signalY, window_len=int(samplerate * 10 / 100))
            signalZ = Prep.smooth(signalZ, window_len=int(samplerate * 10 / 100))

            subplot_count = subplot_count + 1
            plt.subplot(num_sub_plots, 3, subplot_count)
            # https: // towardsdatascience.com / histograms - and -density - plots - in -python - f6bda88f5ac0
            sns.distplot(signalX, hist=True, kde=True, bins=num_bins, color='r')
            plt.title(sett + '_x')
            plt.ylim([0, 0.5])
            plt.xlim([-10, 10])
            plt.grid(True)

            subplot_count = subplot_count + 1
            plt.subplot(num_sub_plots, 3, subplot_count)
            sns.distplot(signalY, hist=True, kde=True, bins=num_bins, color='g')
            # sns.distplot(signalZ, hist=True, kde=True,bins=num_bins)
            plt.title(sett + '_y')
            plt.ylim([0, 0.5])
            plt.xlim([-10, 10])
            plt.grid(True)

            subplot_count = subplot_count + 1
            plt.subplot(num_sub_plots, 3, subplot_count)
            sns.distplot(signalZ, hist=True, kde=True, bins=num_bins, color='b')
            plt.title(sett + '_z')
            plt.ylim([0, 0.5])
            plt.xlim([-10, 10])
            plt.grid(True)

            plt.subplots_adjust(top=0.96, bottom=0.05, left=0.10, right=0.96, hspace=1.0)

        # Sliding window based cutting of the signal
        framesX = sigproc.framesig(signalX, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
        framesY = sigproc.framesig(signalY, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
        framesZ = sigproc.framesig(signalZ, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
        # The  sigproc.framesig function fills zeroes in the last unavailable points
        # we dont want that, so cutting of the last frame
        # some of the begining and end frames are
        # print('framesX',framesX.shape)
        # Removing the last frame because the sigproc adds zeros in the last frame
        framesX = framesX[:-1, :]
        print('working on the imitator ', imt, ' GCAT:', item, 'Setting: ', sett)
        print('framesX', framesX.shape)
        subplot_count2 = 0
        num_sub_plots = framesX.shape[0]
        plt.figure(imt + item + 'data', figsize=(12, 6))
        for frameX, frameY, frameZ in zip(framesX, framesY, framesZ):
            subplot_count2 = subplot_count2 + 1
            plt.subplot(num_sub_plots, 3, subplot_count)
            peaks, _ = signal.find_peaks(frameX, height=1, distance=int(samplerate / 5))
            df = pd.DataFrame(frameX[peaks[0]:]).diff()
            plt.plot(df)

        plt.show()
