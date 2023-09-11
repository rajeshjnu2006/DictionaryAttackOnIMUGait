## THIS SCRIPT HELPS PLOT USERWISE ERROR FOR A CHOSE SENSOR COMBINATION

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
ZeroEffort = pd.DataFrame(columns=['sensors', 'classifier', 'user', 'zero_far', 'zero_frr', 'zero_hter', 'tr_hter'])
########################################################################################################################
zcounter = 0
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
            # Getting the row which has max sfars...# the following shall give one or more rows, at least one
            ze_best = zero_effort_df.sort_values(by='far', ascending=False).iloc[0,:]  # No need to do this, just following the pattern of sorting and picking the best
            ZeroEffort.loc[zcounter] = [sensor_dict[sensor_comb], classifier_dict[cls], ze_best.user, ze_best.far, ze_best.frr, ze_best.hter,
                                        ze_best.refhter]
            zcounter = zcounter + 1

########################################################################################################################
# Prepare the matrix, sensors vs. classifier for FAR, FRR, HTER
ZeroEffortFAR = pd.DataFrame(columns=sensor_name_list)
ZeroEffortFRR = pd.DataFrame(columns=sensor_name_list)
ZeroEffortHTER = pd.DataFrame(columns=sensor_name_list)

for sensors in sensor_name_list:
    temp1 = ZeroEffort[ZeroEffort.sensors == sensors].groupby('classifier', sort=False).mean()
    ZeroEffortFAR[sensors] = temp1['zero_far']
    ZeroEffortFRR[sensors] = temp1['zero_frr']
    ZeroEffortHTER[sensors] = temp1['zero_hter']

# Computing and adding average column
ZeroEffortFAR['Average'] = ZeroEffortFAR.mean(axis=1)
ZeroEffortFRR['Average'] = ZeroEffortFRR.mean(axis=1)
ZeroEffortHTER['Average'] = ZeroEffortHTER.mean(axis=1)

# Changinf errors to percentage
ZeroEffortFAR = ZeroEffortFAR.replace(1, .99)*100
ZeroEffortFRR = ZeroEffortFRR.replace(1, .99)*100
ZeroEffortHTER = ZeroEffortHTER.replace(1, .99)*100
# Sorting as per the mean percent error
ZeroEffortFAR = ZeroEffortFAR.sort_values(by='Average')
ZeroEffortFRR = ZeroEffortFRR.sort_values(by='Average')
ZeroEffortHTER = ZeroEffortHTER.sort_values(by='Average')
# Adding a new row average for the HTER for comparision purpose.!
# Adding for FAr and FRR as well .. to make the comparision easy
ZeroEffortFAR.loc['Average'] = ZeroEffortFAR.mean(axis=0)
ZeroEffortFRR.loc['Average'] = ZeroEffortFRR.mean(axis=0)
ZeroEffortHTER.loc['Average'] = ZeroEffortHTER.mean(axis=0)


########################################################################################################################
########################################################################################################################
sns.set(font_scale=0.9)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row',figsize=(10, 6))
cm = sns.cubehelix_palette(6)
p1 = sns.heatmap(ZeroEffortFAR, annot=True, cmap=cm, ax=ax1, annot_kws={"size":9}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=True, cbar_kws = dict(use_gridspec=False,location="top"))
plt.xlabel("Sensors")
plt.ylabel("Classifiers")
ax1.set_title('FAR under zero-effort attack')

p2 = sns.heatmap(ZeroEffortFRR, annot=True, cmap=cm, ax=ax2, annot_kws={"size":9}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=False)
plt.xlabel("Sensors")
plt.ylabel("Classifiers")
ax2.set_title('FRR under zero-effort attack')
p3 = sns.heatmap(ZeroEffortHTER, annot=True, cmap=cm, ax=ax3, annot_kws={"size":9}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=False)
plt.xlabel("Sensors")
plt.ylabel("Classifiers")
plt.xticks(np.linspace(0.5,16.5,16, endpoint=False),sensor_name_list+['Average'], rotation = 'vertical')
ax3.set_title('HTER under zero-effort attack')
plt.subplots_adjust(left=0.09, bottom=0.17, right=.98, top=0.80, wspace=0.20, hspace=0.30)
plt.tight_layout()
plt.show()
