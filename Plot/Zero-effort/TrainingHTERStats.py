## THIS SCRIPT HELPS PLOT USERWISE ERROR FOR A CHOSEN SENSOR COMBINATION
## JUST CHANGE THE SENSOR_DICT TO GET THE DESIRED RESULTS FOR DESIRED SENSOR COMBINATION
## CURRENTLY WE SHALL BE PLOTTING ONLY TWO SENSOR COMB, ACC and A+G+R which achieved the best HTER IN
## THE INDVIDUAL AND FUSION CATEGORIES
## IT ADDS AN AVERAGE OF AVERAGE COLUMNS, and PLOTS A HEATMAP THE ROWS OF WHICH ARE SORTED BASED ON THE
## AVERAGE OF SENSORS


import os
import sys

import pandas as pd

from Preprocess import Utilities

sys.path.insert(0, os.path.dirname(os.getcwd()))
from GlobalParameters import DefaultParam
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

########################################################################################################################
sense_comb_list = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
sensor_dict = dict(zip(sense_comb_list, Utilities.getallcombinations_sortened(sense_comb_list)))

sense_comb_list = [('LAcc',), ('LAcc', 'Gyr', 'RVec')]
sensor_dict = dict(zip(sense_comb_list, [r"'a'",r"'a+g+r'"]))

# Individual best
# sense_comb_list = [('LAcc',)]
# sensor_dict = dict(zip(sense_comb_list, [r"'a'"]))
# Comb best
# sense_comb_list = [('LAcc', 'Gyr', 'RVec')]
# sensor_dict = dict(zip(sense_comb_list, [r"'a+g+r'"]))

sensor_name_list = []
for key in sensor_dict:
    sensor_name_list.append(sensor_dict[key])
#
########################################################################################################################
# creating an ordered user list for sanity
user_list = []
for i in range(1, 56):
    user_list.append('User' + str(i))

########################################################################################################################
ZeroEffort = pd.DataFrame(columns=['sensor_comb','classifier', 'user', 'zero_far', 'zero_frr', 'zero_hter', 'tr_hter'])
########################################################################################################################
zcounter = 0
classifier_dict = {'../ResultFiles/kNNZDHcombined.csv': 'kNN', '../ResultFiles/LRegZDHcombined.csv': 'LReg', '../ResultFiles/MLPZDHcombined.csv': 'MLP', '../ResultFiles/RanForZDHcombined.csv': 'RanFor', '../ResultFiles/SVMZDHcombined.csv': 'SVM'}
cls_name_list = []
for key in classifier_dict:
    cls_name_list.append(classifier_dict[key])
########################################################################################################################
for cls in classifier_dict:
    results_df = pd.read_csv(cls, dtype={'sensors': str}, index_col=0)
    for si, sensor_comb in enumerate(sensor_dict):
        curr_comb_df = results_df.loc[results_df['sensors'] == str(sensor_comb)]  # slicing the dataframe for the
        # current
        # user_list = ['User1', 'User2']
        for user in user_list:
            # Slicing the dataframe for the current user and current sensor_comb
            curr_user_df = curr_comb_df.loc[curr_comb_df['user'] == user]
            zero_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'zero-effort']
            # Getting the row which has max sfars...# the following shall give one or more rows, at least one
            ze_best = zero_effort_df.sort_values(by='far', ascending=False).iloc[0,:]  # No need to do this, just
            # following the pattern of sorting and picking the best
            ZeroEffort.loc[zcounter] = [sensor_name_list[si], classifier_dict[cls], ze_best.user, ze_best.far, ze_best.frr, ze_best.hter,
                                        ze_best.refhter]
            zcounter = zcounter + 1

########################################################################################################################

print(ZeroEffort.to_string())
ZeroEffortFAR = pd.DataFrame(columns=user_list)
ZeroEffortFRR = pd.DataFrame(columns=user_list)
ZeroEffortHTER = pd.DataFrame(columns=user_list)
ZeroEffortTrHTER_A = pd.DataFrame(columns=user_list)
ZeroEffortTrHTER_AGR = pd.DataFrame(columns=user_list)

row_count = 0
for classifier in cls_name_list:
    curr_classifier = ZeroEffort[ZeroEffort.classifier == classifier]

    LAcc = curr_classifier[ZeroEffort.sensor_comb == sensor_name_list[0]]
    Fused = curr_classifier[ZeroEffort.sensor_comb == sensor_name_list[1]]

    ZeroEffortTrHTER_A.loc[classifier] = LAcc['tr_hter'].values
    ZeroEffortTrHTER_AGR.loc[classifier] = Fused['tr_hter'].values


# Computing and adding average column

ZeroEffortTrHTER_A['Average'] = ZeroEffortTrHTER_A.mean(axis=1)
ZeroEffortTrHTER_AGR['Average'] = ZeroEffortTrHTER_AGR.mean(axis=1)

print('ZeroEffortTrHTER_A')
print(ZeroEffortTrHTER_A.to_string())
print('ZeroEffortTrHTER_AGR')
print(ZeroEffortTrHTER_AGR.to_string())
# Changing errors to percentage

ZeroEffortTrHTER_A = ZeroEffortTrHTER_A*100
ZeroEffortTrHTER_AGR = ZeroEffortTrHTER_AGR*100

# Sorting as per the mean percent error
ZeroEffortTrHTER_A = ZeroEffortTrHTER_A.sort_values(by='Average')
ZeroEffortTrHTER_AGR = ZeroEffortTrHTER_AGR.sort_values(by='Average')


sns.set(font_scale=0.9)
fig, ax = plt.subplots(2, 1, sharex='col', sharey='row',figsize=(10, 4))
cm = sns.cubehelix_palette(6)

p4 = sns.heatmap(ZeroEffortTrHTER_A, annot=True, cmap=cm, ax=ax[0], annot_kws={"size":8}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=True, cbar_kws = dict(use_gridspec=False,location="top"))
plt.xlabel("Users")
plt.ylabel("Classifiers")
ax[0].set_title('Training HTER for '+str(sensor_name_list[0]))

p5 = sns.heatmap(ZeroEffortTrHTER_AGR, annot=True, cmap=cm, ax=ax[1], annot_kws={"size":8}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=False)#True, cbar_kws = dict(use_gridspec=False,location="top"))
plt.xlabel("Users")
plt.ylabel("Classifiers")
ax[1].set_title('Training HTER for '+str(sensor_name_list[1]))
plt.xticks(np.linspace(0.5, 56.5, 56, endpoint=False), user_list+['Average'])
plt.subplots_adjust(left=0.08, bottom=0.17, right=.98, top=0.80, wspace=0.20, hspace=0.20)

plt.tight_layout()
plt.show()
