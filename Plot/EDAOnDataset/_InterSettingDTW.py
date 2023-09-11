### https://nipunbatra.github.io/blog/2014/dtw.html
# https://www.youtube.com/watch?v=_K1OsqCicBY watch at 4:30 to see how value are computed and 5:30 to see the smallest path
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
cut_points = 230  # Taking away 5 seconds of data
total_points = 2070
num_frames_to_compare = 5
list_of_gcats = ['slength', 'swidth', 'tlift','speed']
imitators = ['Imitator2','Imitator3']
# For each imitator
for imt in imitators:  ## Traversing over all imitators
    curr_imp_folder = os.path.join(imdb_path, imt)
    all_gcat_settings = os.listdir(curr_imp_folder)
    list_of_speed_settings = []
    list_of_slength_settings = []
    list_of_swidth_settings = []
    list_of_tlift_settings = []

    # for all gcat settings
    for setting in all_gcat_settings:
        if 'speed' in setting:
            list_of_speed_settings.append(setting)
        if 'slength' in setting:
            list_of_slength_settings.append(setting)
        if 'swidth' in setting:
            list_of_swidth_settings.append(setting)
        if 'tlift' in setting:
            list_of_tlift_settings.append(setting)

    # for all gcats
    for gcat in list_of_gcats:
        # Manual ordering to ensure that the gcat settings appear in order while plotting
        if gcat == 'speed':
            # Using only odd indices, speed was perfectly ordered to so no manual ordering
            # Also taking only odd stages for the sake of plotting
            list_of_settings = list_of_speed_settings[0::2]
        elif gcat == 'slength':
            list_of_settings = ['slength_short', 'slength_normal', 'slength_long', 'slength_longer']
        elif gcat == 'swidth':
            list_of_settings = ['swidth_close', 'swidth_normal', 'swidth_wide', 'swidth_wider']
        else:
            list_of_settings = ['tlift_back', 'tlift_normal', 'tlift_front', 'tlift_up']
        #############################
        # print('The current GCAT {} and corresponding settings are {}'.format(gcat, list_of_settings))
        num_settings_cgcat = len(list_of_settings)

        # ### Ensuring that that number is even so they fit on the plot
        # if (num_settings_cgcat & 1):
        #     list_of_settings = list_of_settings[:-1]
        #     num_settings_cgcat = num_settings_cgcat - 1

        num_subplots = num_settings_cgcat * num_settings_cgcat
        # counter for the plots
        plot_counter = 0
        # creating a figure
        plt.figure("{}_{}_{}".format(imt,gcat,'DTW_x'), figsize=(12, 6))

        df_dtw_dist = pd.DataFrame(columns=list_of_settings)

        for first in list_of_settings:
            first_path = os.path.join(curr_imp_folder, first,attack_sensor)
            # Computing the feature vector for the current imitator at current setting
            first_raw_data = pd.read_csv(first_path, sep=',', header=None, names=DefaultParam.COLUMN_NAMES)

            ## Cutting useless data points from the begin and end
            firstX = first_raw_data['x'][cut_points:total_points]
            firstY = first_raw_data['y'][cut_points:total_points]
            firstZ = first_raw_data['z'][cut_points:total_points]

            # Smoothing
            firstX = Prep.smooth(firstX, window_len=int(samplerate * 10 / 100))
            firstY = Prep.smooth(firstY, window_len=int(samplerate * 10 / 100))
            firstZ = Prep.smooth(firstZ, window_len=int(samplerate * 10 / 100))

            # Sliding window based cutting of the signal
            # ParameterskNN.WINSTEP = 4
            framesInFirstX = sigproc.framesig(firstX, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            framesInFirstY = sigproc.framesig(firstY, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
            framesInFirstZ = sigproc.framesig(firstZ, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)

            # random.seed(ParameterskNN.RANDOM_SEED)
            rfids_first = random.sample(range(0, framesInFirstX.shape[0]-1), num_frames_to_compare)
            chosenFirstX = framesInFirstX[rfids_first, :]
            chosenFirstY = framesInFirstY[rfids_first, :]
            chosenFirstZ = framesInFirstZ[rfids_first, :]
            avg_dist_list = []
            for second in list_of_settings:
                plot_counter = plot_counter + 1
                plt.subplot(int(num_subplots / num_settings_cgcat), num_settings_cgcat, plot_counter)
                plt.title('{} to {}'.format(first.replace(gcat,''), second.replace(gcat,'')))
                # if first != second:
                # print('working on settings {} and {}'.format(first,second))
                second_path = os.path.join(curr_imp_folder, second, attack_sensor)
                # Computing the feature vector for the current imitator at current setting
                second_raw_data = pd.read_csv(second_path, sep=',', header=None, names=DefaultParam.COLUMN_NAMES)

                ## Cutting useless data points from the begin and end
                secondX = second_raw_data['x'][cut_points:total_points]
                secondY = second_raw_data['y'][cut_points:total_points]
                secondZ = second_raw_data['z'][cut_points:total_points]
                # Smoothing
                secondX = Prep.smooth(secondX, window_len=int(samplerate * 10 / 100))
                secondY = Prep.smooth(secondY, window_len=int(samplerate * 10 / 100))
                secondZ = Prep.smooth(secondZ, window_len=int(samplerate * 10 / 100))
                # Sliding window based cutting of the signal
                # ParameterskNN.WINSTEP = 4
                framesInSecondX = sigproc.framesig(secondX, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
                framesInSecondY = sigproc.framesig(secondY, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
                framesInSecondZ = sigproc.framesig(secondZ, DefaultParam.WINLENGTH * samplerate, DefaultParam.WINSTEP * samplerate)
                # random.seed(ParameterskNN.RANDOM_SEED)
                rfids_second = random.sample(range(0, framesInSecondX.shape[0] - 1), num_frames_to_compare)
                chosenSecondX = framesInSecondX[rfids_second, :]
                chosenSecondY = framesInSecondY[rfids_second, :]
                chosenSecondZ = framesInSecondZ[rfids_second, :]
                color = iter(cm.rainbow(np.linspace(0, 1, num_frames_to_compare)))
                dist_list = []
                for ind1, frameX1 in enumerate(chosenFirstX):
                    for ind2, frameX2 in enumerate(chosenSecondX):
                        # if ind1 != ind2:
                        distance, path = fastdtw(frameX1, frameX2, dist=euclidean)
                        dist_list.append(distance)
                        # print("DTW distance [(xi-yi)^2] between frame [{}] of {},and [{}] of {}".format(ind1,first,ind2,second),int(distance*distance))
                        x, y = zip(*path)
                        # plt.plot(x, y, '-')
                        # plt.ylim(0,400)
                        # plt.xlim(0,400)
                        # plt.xticks(np.arange(0, 400, step=100))
                        # plt.yticks(np.arange(0, 400, step=100))
                        # plt.xlabel('windows_'+first.replace(gcat,''))
                        # plt.ylabel('windows_'+second.replace(gcat,''))
                # plt.grid(True)
                print("{}: Average DTW dist b/w frames of {} and {} is {}".format(imt, first, second,int(stat.mean(dist_list))))
            avg_dist_list.append(stat.mean(dist_list))
                # plt.subplots_adjust(top=0.91, bottom=0.08, left=0.10, right=0.96, hspace=0.9, wspace=0.8)
            df_dtw_dist.loc[first] = [avg_dist_list]
        # plt.show()
        print(df_dtw_dist)
