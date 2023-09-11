import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import os
from Preprocess import Utilities
from Preprocess import Prep
from python_speech_features import sigproc
from scipy import signal
from GlobalParameters import DefaultParam as DP
path = '../../Storage/RawDataGenuine'
samplerate = 46
num_bins = 80  # Acc varies between -10g to 10 g
# Taking about 40 sec of data points | (46*40) 1840 data points
begin_end_remove_data_points = 500  # Taking away 5 seconds of data
total_data_points = 2070
num_user = len(os.listdir(path))
random.seed(1985)
num_plots = 9
yplotrange = 45
random_list = random.sample(range(30, num_user), num_plots)
sorted_user_list = ['User'+str(i) for i in random_list]
num_plots = len(sorted_user_list)

# ###################Raw plots##############################
# plt.figure('Raw data plots')
# plot_counter = 0
# for user in sorted_user_list:
#     data_file_acc = os.path.join(path,user,'Training', 'clean_LAcc.txt')
#     # print(data_file_acc)
#     RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.NEW_COLUMN_NAMES)
#
#     # The old data had different sampling rates.. here computing the sampling rate dynamically
#     if user in DP.OLD_USER_LIST:
#         RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.OLD_COLUMN_NAMES)
#         samplerate = Utilities.get_sampling_rate(RAW_DATA)
#     else:
#         RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.NEW_COLUMN_NAMES)
#         samplerate = DP.NEW_SAMPLING_RATE
#     print('sampling rate for ' + data_file_acc + ': ' + str(samplerate))
#
#     signalX = RAW_DATA['x']
#     signalY = RAW_DATA['y']
#     signalZ = RAW_DATA['z']
#
#     signalX = signalX[begin_end_remove_data_points:total_data_points]
#     signalY = signalY[begin_end_remove_data_points:total_data_points]
#     signalZ = signalZ[begin_end_remove_data_points:total_data_points]
#
#     # Sliding window based cutting of the signal
#     # ParameterskNN.WINSTEP = 4
#     framesX = sigproc.framesig(signalX, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
#     framesY = sigproc.framesig(signalY, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
#     framesZ = sigproc.framesig(signalZ, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
#     # The  sigproc.framesig function fills zeroes in the last unavailable points
#     # we dont want that, so cutting of the last frame
#     # some of the begining and end frames are
#     # print('framesX',framesX.shape)
#     # Removing the first and last frame because the sigproc adds zeros in the last frame
#     framesX = framesX[0:-1, :]
#     framesY = framesY[0:-1, :]
#     framesZ = framesZ[0:-1, :]
#
#     plot_counter = plot_counter + 1
#     plt.subplot(int(num_plots / 3), 3, plot_counter)
#
#     color = iter(cm.rainbow(np.linspace(0, 1, framesX.shape[0])))
#     for frameX, frameY, frameZ in zip(framesX, framesY,framesZ):
#         frameX = frameX +20
#         ax = sns.lineplot(x=list(range(0,len(frameX))), y=frameX)
#         # frameY = frameY
#         sns.lineplot(x=list(range(0,len(frameY))), y=frameY, ax=ax)
#         frameZ = frameZ - 20
#         sns.lineplot(x=list(range(0,len(frameZ))), y=frameZ, ax=ax)
#
#         plt.ylim([-yplotrange, yplotrange])
#         plt.xlabel('Time')
#         plt.ylabel('Acc(m/s^2)')
#         plt.title(user,fontsize=10)
#         break
#     plt.grid(True)
#     plt.gcf().set_size_inches(9, 5)
#     plt.subplots_adjust(top=0.91, bottom=0.08, left=0.10, right=0.96, hspace=0.4, wspace=0.4)
# plt.legend(['x', 'y', 'z'], ncol=3,framealpha=0.0, fontsize=8)
# plt.tight_layout()
# # plt.show()

###################Smooth plots##############################
plt.figure('Smoothed data plots')
plot_counter = 0
for user in sorted_user_list:
    data_file_acc = os.path.join(path,user,'Training', 'clean_LAcc.txt')
    print(data_file_acc)
    RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.NEW_COLUMN_NAMES)

    # The old data had different sampling rates.. here computing the sampling rate dynamically
    if user in DP.OLD_USER_LIST:
        RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.OLD_COLUMN_NAMES)
        samplerate = Utilities.get_sampling_rate(RAW_DATA)
    else:
        RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.NEW_COLUMN_NAMES)
        samplerate = DP.NEW_SAMPLING_RATE
    print('sampling rate for ' + data_file_acc + ': ' + str(samplerate))

    signalX = RAW_DATA['x']
    signalY = RAW_DATA['y']
    signalZ = RAW_DATA['z']

    signalX = signalX[begin_end_remove_data_points:total_data_points]
    signalY = signalY[begin_end_remove_data_points:total_data_points]
    signalZ = signalZ[begin_end_remove_data_points:total_data_points]

    signalX = Prep.smooth(signalX, window_len=int(samplerate * 5 / 100))
    signalY = Prep.smooth(signalY, window_len=int(samplerate * 5 / 100))
    signalZ = Prep.smooth(signalZ, window_len=int(samplerate * 5 / 100))

    # Sliding window based cutting of the signal
    # ParameterskNN.WINSTEP = 4
    framesX = sigproc.framesig(signalX, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
    framesY = sigproc.framesig(signalY, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
    framesZ = sigproc.framesig(signalZ, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
    # The  sigproc.framesig function fills zeroes in the last unavailable points
    # we dont want that, so cutting of the last frame
    # some of the begining and end frames are
    # print('framesX',framesX.shape)
    # Removing the first and last frame because the sigproc adds zeros in the last frame
    framesX = framesX[0:-1, :]
    framesY = framesY[0:-1, :]
    framesZ = framesZ[0:-1, :]

    plot_counter = plot_counter + 1
    plt.subplot(int(num_plots / 3), 3, plot_counter)

    for frameX, frameY, frameZ in zip(framesX, framesY,framesZ):
        frameX = frameX + 20
        ax = sns.lineplot(x=list(range(0, len(frameX))), y=frameX, lw=1)
        # frameY = frameY
        sns.lineplot(x=list(range(0, len(frameY))), y=frameY, ax=ax, lw=1)
        frameZ = frameZ - 20
        sns.lineplot(x=list(range(0, len(frameZ))), y=frameZ, ax=ax, lw=1)

        plt.ylim([-yplotrange, yplotrange])
        if plot_counter ==7 or plot_counter ==8 or plot_counter ==9:
            plt.xlabel('Time')
        if plot_counter ==1 or plot_counter ==4 or plot_counter ==7:
            plt.ylabel('Acc($m/s^2$)')
        plt.yticks(list(np.linspace(start=-yplotrange, stop=yplotrange, num=4)))
        plt.xticks(list(np.linspace(start=0, stop=400, num=5)))
        # plt.legend(['x', 'y', 'z'], ncol=3)
        plt.title(user,fontsize=10)
        plt.legend(['x', 'y', 'z'], ncol=3, framealpha=0.0, fontsize=8)
        break
    plt.grid(True)
    plt.gcf().set_size_inches(9, 6)
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.10, right=0.96, hspace=0.96, wspace=0.45)
plt.tight_layout()
# plt.show()

###################Density plots##############################
plt.figure('Probability density plots')
plot_counter = 0
for user in sorted_user_list:
    data_file_acc = os.path.join(path,user,'Training', 'clean_LAcc.txt')
    # print(data_file_acc)
    RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.NEW_COLUMN_NAMES)

    # The old data had different sampling rates.. here computing the sampling rate dynamically
    if user in DP.OLD_USER_LIST:
        RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.OLD_COLUMN_NAMES)
        samplerate = Utilities.get_sampling_rate(RAW_DATA)
    else:
        RAW_DATA = pd.read_csv(data_file_acc, sep=',', header=None, names=DP.NEW_COLUMN_NAMES)
        samplerate = DP.NEW_SAMPLING_RATE
    print('sampling rate for ' + data_file_acc + ': ' + str(samplerate))

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
    framesX = sigproc.framesig(signalX, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
    framesY = sigproc.framesig(signalY, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
    framesZ = sigproc.framesig(signalZ, DP.WINLENGTH * samplerate, DP.WINSTEP * samplerate)
    # The  sigproc.framesig function fills zeroes in the last unavailable points
    # we dont want that, so cutting of the last frame
    # some of the begining and end frames are
    # print('framesX',framesX.shape)
    # Removing the first and last frame because the sigproc adds zeros in the last frame
    framesX = framesX[0:-1, :]
    framesY = framesY[0:-1, :]
    framesZ = framesZ[0:-1, :]

    plot_counter = plot_counter + 1
    plt.subplot(int(num_plots / 3), 3, plot_counter)

    color = iter(cm.rainbow(np.linspace(0, 1, framesX.shape[0])))
    for frameX in framesX:
        peaks, _ = signal.find_peaks(frameX, height=1, distance=int(samplerate / 5))
        # sns.lineplot(x=list(range(0,len(frameX))), y=frameX)
        sns.distplot(frameX, rug=False, hist=True, kde=True, bins=num_bins, norm_hist=True, kde_kws={"lw": 1}, hist_kws={"linewidth": 1})
        plt.ylim([0, 0.35])
        plt.yticks(list(np.linspace(start=0,stop=0.3,num=5)))
        plt.xticks(list(np.linspace(start=-15,stop=15,num=6)))
        plt.xlim([-15, 15])
        if plot_counter ==1 or plot_counter ==4 or plot_counter ==7:
            plt.ylabel('Prob. density')

        if plot_counter ==7 or plot_counter ==8 or plot_counter ==9:
            plt.xlabel('$Acc_X(m/s^2$)')

        # plt.legend(['x','y','z'])
        plt.title(user,fontsize=10)

    plt.grid(True)
    plt.gcf().set_size_inches(9, 6)
    plt.subplots_adjust(top=0.91, bottom=0.08, left=0.10, right=0.96, hspace=0.62, wspace=0.45)
plt.legend(fontsize=8,framealpha=0.0)
plt.tight_layout()
plt.show()