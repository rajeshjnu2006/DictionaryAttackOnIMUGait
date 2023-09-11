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
imdb_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Storage','PreprocessedProfilingAnonymous')
imitators = os.listdir(imdb_path)
exclude_imt_list = ['Imitator4', 'Imitator6']
for imitator in exclude_imt_list:
    imitators.remove(imitator)

attack_sensor = 'clean_LAcc.txt'
samplerate = 46
num_bins = 20  # Acc varies between -10g to 10 g
# Taking about 40 sec of data points | (46*40) 1840 data points
begin_end_remove_data_points = 230  # Taking away 5 seconds of data
total_data_points = 2070
gcat_list = ['speed','slength', 'swidth', 'tlift']
imitators = ['Imitator3','Imitator7','Imitator9']

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

    for gcat in gcat_list:
        if gcat == 'speed':
            # Using only odd indices, speed was perfectly ordered to so no manual ordering
            set_list = speed_sett_list[0::2]
        elif gcat == 'slength':
            set_list = ['slength_short', 'slength_normal', 'slength_long', 'slength_longer']
        elif gcat == 'swidth':
            set_list = ['swidth_close', 'swidth_normal', 'swidth_wide', 'swidth_wider']
        else:
            set_list = ['tlift_back', 'tlift_normal', 'tlift_front', 'tlift_up']

        #############################
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

            # Sliding window based cutting of the signal
            framesX = sigproc.framesig(signalX, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            framesY = sigproc.framesig(signalY, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            framesZ = sigproc.framesig(signalZ, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            # The  sigproc.framesig function fills zeroes in the last unavailable points
            # we dont want that, so cutting of the last frame
            # some of the begining and end frames are
            # print('framesX',framesX.shape)
            # Removing the first and last frame because the sigproc adds zeros in the last frame
            framesX = framesX[0:8, :]
            framesY = framesY[0:8, :]
            framesZ = framesZ[0:8, :]

            print('working on the imitator ', imt, ' GCAT:', gcat, 'Setting: ', sett)
            print('framesX', framesX.shape)
            frame_count = 0
            number_of_frame_rows = int(framesX.shape[0]/2)
            print('number_of_frames',number_of_frame_rows)
            plt.figure('{}_{}_({})'.format(imt, sett,'x'), figsize=(12, 6))
            for frameX in framesX:
                frame_count = frame_count + 1
                plt.subplot(number_of_frame_rows, 2, frame_count)
                peaks, _ = signal.find_peaks(frameX, height=1, distance=int(samplerate / 5))
                df = pd.DataFrame(frameX[peaks[0]:])
                plt.plot(df)
                plt.ylim([-10, 10])
                plt.xlim([0, 400])
                plt.grid(True)
                plt.subplots_adjust(top=0.96, bottom=0.05, left=0.10, right=0.96, hspace=1.0)

            plt.figure('{}_{}_({})'.format(imt, sett,'y'), figsize=(12, 6))
            frame_count = 0
            for frameY in framesY:
                frame_count = frame_count + 1
                plt.subplot(number_of_frame_rows, 2, frame_count)
                peaks, _ = signal.find_peaks(frameY, height=1, distance=int(samplerate / 5))
                df = pd.DataFrame(frameY[peaks[0]:])
                plt.plot(df)
                plt.ylim([-10, 10])
                plt.xlim([0, 400])
                plt.grid(True)
                plt.subplots_adjust(top=0.96, bottom=0.05, left=0.10, right=0.96, hspace=1.0)

            plt.figure('{}_{}_({})'.format(imt, sett,'z'), figsize=(12, 6))
            frame_count = 0
            for frameZ in framesZ:
                frame_count = frame_count + 1
                plt.subplot(number_of_frame_rows, 2, frame_count)
                peaks, _ = signal.find_peaks(frameZ, height=1, distance=int(samplerate / 5))
                df = pd.DataFrame(frameZ[peaks[0]:])
                plt.plot(df)
                plt.ylim([-10, 10])
                plt.xlim([0, 400])
                plt.grid(True)
                plt.subplots_adjust(top=0.96, bottom=0.05, left=0.10, right=0.96, hspace=1.0)

        plt.show()
