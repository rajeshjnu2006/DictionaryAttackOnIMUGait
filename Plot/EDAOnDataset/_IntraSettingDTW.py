### https://nipunbatra.github.io/blog/2014/dtw.html
# https://www.youtube.com/watch?v=_K1OsqCicBY watch at 4:30 to see how value are computed and 5:30 to see the smallest path
import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
import random
import statistics as stat
import numpy as np
import seaborn as sns
import os
import sys
# Adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from GlobalParameters import DefaultParam
from Preprocess import Prep
from python_speech_features import sigproc
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

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
imitators = ['Imitator9']

########Creatuing some useful variables
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
            # Also taking only odd stages for the sake of plotting
            set_list = speed_sett_list[0::2]
        elif gcat == 'slength':
            set_list = ['slength_short', 'slength_normal', 'slength_long', 'slength_longer']
        elif gcat == 'swidth':
            set_list = ['swidth_close', 'swidth_normal', 'swidth_wide', 'swidth_wider']
        else:
            set_list = ['tlift_back', 'tlift_normal', 'tlift_front', 'tlift_up']

        #############################
        print('Set_list:', set_list)
        number_of_plots = len(set_list)
        if (number_of_plots & 1):
            set_list = set_list[:-1]
            number_of_plots = number_of_plots - 1
        plot_counter = 0
        plt.figure("{}_{}_{}".format(imt,gcat,'dtwdistance'), figsize=(12, 6))
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
            # ParameterskNN.WINSTEP = 4
            framesX = sigproc.framesig(signalX, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            framesY = sigproc.framesig(signalY, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            framesZ = sigproc.framesig(signalZ, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            # The  sigproc.framesig function fills zeroes in the last unavailable points
            # we dont want that, so cutting of the last frame
            # some of the begining and end frames are
            # print('framesX',framesX.shape)
            # Removing the first and last frame because the sigproc adds zeros in the last frame
            number_of_frames = framesX.shape[0]
            num_frame_to_plot = 5
            random.seed(42)
            random_frame_ind = random.sample(range(0, number_of_frames-1),num_frame_to_plot)
            framesX = framesX[random_frame_ind, :]
            framesY = framesY[random_frame_ind, :]
            framesZ = framesZ[random_frame_ind, :]

            # print('working on the imitator ', imt, ' GCAT:', gcat, 'Setting: ', sett)
            # print('framesX', framesX.shape)
            plot_counter = plot_counter + 1

            plt.subplot(int(number_of_plots/2), 2, plot_counter)
            plt.title('{}_{}_({})'.format(imt, sett,'x'), )

            color = iter(cm.rainbow(np.linspace(0, 1, framesX.shape[0])))
            dist_list = []
            for ind1, frameX1 in enumerate(framesX):
                for ind2, frameX2 in enumerate(framesX):
                    distance, path = fastdtw(frameX1, frameX2, dist=euclidean)
                    dist_list.append(distance)
                    # print("DTW distance between frame [{}] of {},and [{}] of {}".format(ind1,sett,ind2,sett),distance)
                    x, y = zip(*path)
                    plt.plot(x, y, '-')
                    plt.ylim([0,400])
                    plt.xlim([0,400])
                    plt.xlabel('frames_i')
                    plt.ylabel('frames_j')
            print("Average DTW dist b/w frames of {} is {}".format(sett, stat.mean(dist_list)))

            plt.grid(True)
            plt.gcf().set_size_inches(15, 8)
            plt.subplots_adjust(top=0.91, bottom=0.08, left=0.10, right=0.96, hspace=0.4, wspace=0.4)
        plt.show()
