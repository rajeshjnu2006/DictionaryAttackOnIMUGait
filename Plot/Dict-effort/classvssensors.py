## THIS SCRIPT HELPS PLOT SENSORWISE-CALSSIFIER WISE DAMAGE
## THIS SCRIPT HELPS PLOT SENSORWISE RESULTS AVERAGED OVER ALL THE USERS
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
sensor_name_list = []
for key in sensor_dict:
    sensor_name_list.append(sensor_dict[key])

# creating an ordered user list for sanity
user_list = []
for i in range(1, 56):
    user_list.append('User' + str(i))

########################################################################################################################
DictionaryEffort = pd.DataFrame(columns=['sensors', 'classifier', 'user', 'attacker', 'zero_far', 'dict_far', 'zero_hter', 'dict_hter'])
########################################################################################################################
dcounter = 0
classifier_dict = {'../ResultFiles/kNNZDHcombined.csv': 'kNN', '../ResultFiles/LRegZDHcombined.csv': 'LReg', '../ResultFiles/MLPZDHcombined.csv': 'MLP', '../ResultFiles/RanForZDHcombined.csv': 'RanFor', '../ResultFiles/SVMZDHcombined.csv': 'SVM'}
cls_name_list = []
for key in classifier_dict:
    cls_name_list.append(classifier_dict[key])
########################################################################################################################
for cls in classifier_dict:
    results_df = pd.read_csv(cls, dtype={'sensors': str}, index_col=0)
    for sensor_comb in sensor_dict:
        curr_comb_df = results_df.loc[results_df['sensors'] == str(sensor_comb)]  # slicing the dataframe for the current
        # user_list = ['User1', 'User2']
        for user in user_list:
            # Slicing the dataframe for the current user and current sensor_comb
            curr_user_df = curr_comb_df.loc[curr_comb_df['user'] == user]

            zero_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'zero-effort']
            dict_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'dict-effort']

            # Getting the row which has max sfars...# the following shall give one or more rows, at least one
            ze_best = zero_effort_df.sort_values(by='far', ascending=False).iloc[0, :]
            de_best = dict_effort_df.sort_values(by='far', ascending=False).iloc[0, :]

            DictionaryEffort.loc[dcounter] = [sensor_dict[sensor_comb], classifier_dict[cls], user, de_best.attacker, ze_best.far,
                                              de_best.far, ze_best.hter, de_best.hter]
            dcounter = dcounter + 1

########################################################################################################################
# Prepare the matrix, sensors vs. classifier for FAR, DFAR, HTER, DHTER
ZeroEffortFAR = pd.DataFrame(columns=sensor_name_list)
DictEffortFAR = pd.DataFrame(columns=sensor_name_list)
ZeroEffortHTER = pd.DataFrame(columns=sensor_name_list)
DictEffortHTER = pd.DataFrame(columns=sensor_name_list)

for sensors in sensor_name_list:
    # Taking the average over all the users, for each sensor
    temp1 = DictionaryEffort[DictionaryEffort.sensors == sensors].groupby('classifier', sort=False).mean()
    ZeroEffortFAR[sensors] = temp1['zero_far']
    DictEffortFAR[sensors] = temp1['dict_far']
    ZeroEffortHTER[sensors] = temp1['zero_hter']
    DictEffortHTER[sensors] = temp1['dict_hter']

# Computing and adding average column
ZeroEffortFAR['Average'] = ZeroEffortFAR.mean(axis=1)
DictEffortFAR['Average'] = DictEffortFAR.mean(axis=1)
ZeroEffortHTER['Average'] = ZeroEffortHTER.mean(axis=1)
DictEffortHTER['Average'] = DictEffortHTER.mean(axis=1)

# Changing errors to percentage
# ZeroEffortFAR = ZeroEffortFAR.replace(1, .99)*100
# DictEffortFAR = DictEffortFAR.replace(1, .99)*100
# ZeroEffortHTER = ZeroEffortHTER.replace(1, .99)*100
# DictEffortHTER = DictEffortHTER.replace(1, .99)*100

# Changing errors to percentage
ZeroEffortFAR = ZeroEffortFAR*100
DictEffortFAR = DictEffortFAR*100
ZeroEffortHTER = ZeroEffortHTER*100
DictEffortHTER = DictEffortHTER*100

# Sorting as per the mean percent error
ZeroEffortFAR = ZeroEffortFAR.sort_values(by='Average')
# Adding a new row average for the HTER for comparision purpose.!
ZeroEffortFAR.loc['Average'] = ZeroEffortFAR.mean(axis=0)

DictEffortFAR = DictEffortFAR.sort_values(by='Average')
DictEffortFAR.loc['Average'] = DictEffortFAR.mean(axis=0)

ZeroEffortHTER = ZeroEffortHTER.sort_values(by='Average')
ZeroEffortHTER.loc['Average'] = ZeroEffortHTER.mean(axis=0)

DictEffortHTER = DictEffortHTER.sort_values(by='Average')
DictEffortHTER.loc['Average'] = DictEffortHTER.mean(axis=0)

########################################################################################################################
sns.set(font_scale=0.9)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='col', sharey='row', figsize=(8, 6))
# cm = sns.cubehelix_palette(rot=-.2)
# cm = sns.light_palette((20, 60, 50), input="husl")
cm = sns.light_palette("xkcd:copper", 12)
p1 = sns.heatmap(ZeroEffortFAR, annot=True, cmap=cm, ax=ax1, annot_kws={"size":9}, vmin=0, vmax=100, fmt='.0f', linewidths=0.1, cbar=False)#, cbar_kws = dict(use_gridspec=False, location="top"))
plt.xlabel("Sensors")
# plt.yticks(np.linspace(0,5,5, endpoint=False), cls_name_list+['Average'], rotation = 'horizontal')
plt.ylabel("Classifiers")
ax1.set_title('FAR under zero-effort attack')

p2 = sns.heatmap(DictEffortFAR, annot=True, cmap=cm, ax=ax2, annot_kws={"size":9}, vmin=0, vmax=100, fmt='.0f', linewidths=0.1, cbar=False)
plt.xlabel("Sensors")
plt.ylabel("Classifiers")
ax2.set_title('FAR under dictionary-effort attack')

p3 = sns.heatmap(ZeroEffortHTER, annot=True, cmap=cm, ax=ax3, annot_kws={"size":9}, vmin=0, vmax=100, fmt='.0f', linewidths=0.1, cbar=False)
plt.xlabel("Sensors")
plt.ylabel("Classifiers")
ax3.set_title('HTER under zero-effort attack')

p4 = sns.heatmap(DictEffortHTER, annot=True, cmap=cm, ax=ax4, annot_kws={"size":9}, vmin=0, vmax=100, fmt='.0f', linewidths=0.1, cbar=False)
plt.xlabel("Sensors")
plt.ylabel("Classifiers")
plt.xticks(np.linspace(0.5,16.5,16, endpoint=False),sensor_name_list+['Average'], rotation = 'vertical')
ax4.set_title('HTER under dictionary-effort attack')

plt.subplots_adjust(left=0.09, bottom=0.15, right=.98, top=0.96, wspace=0.20, hspace=0.30)
# plt.tight_layout()
plt.show()
