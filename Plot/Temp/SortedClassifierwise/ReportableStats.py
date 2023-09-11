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

# Overrriding this for individual sensors
# sense_comb_list = [('LAcc',)]
# sensor_dict = dict(zip(sense_comb_list, ['a']))

# creating an ordered user list for sanity
user_list = []
for i in range(1, 56):
    user_list.append('User' + str(i))

########################################################################################################################
ZeroEffort = pd.DataFrame(columns=['user', 'sensors', 'classifier', 'far', 'frr', 'hter'])
DictionaryEffort = pd.DataFrame(columns=['user', 'sensors', 'classifier', 'far', 'frr', 'hter'])
HighEffort = pd.DataFrame(columns=['user', 'sensors', 'classifier', 'far', 'frr', 'hter'])

########################################################################################################################
zcounter = 0
dcounter = 0
hcounter = 0
classifiers = {'../kNNZDHcombined.csv':'kNN', '../LRegZDHcombined.csv':'LReg', '../MLPZDHcombined.csv':'MLP','../RanForZDHcombined.csv':'RanFor','../SVMZDHcombined.csv':'SVM'}

for cls in classifiers:
    results_df = pd.read_csv(cls, dtype={'sensors': str}, index_col=0)
    for sensor_comb in sensor_dict:
        curr_comb_df = results_df.loc[results_df['sensors'] == str(sensor_comb)]  # slicing the dataframe for the current
        # sensor comb
        # Traversing all the users for the current sensor
        # ordered_user_list = ['User1', 'User2']
        for user in user_list:
            # print(f'running for {sensor_comb} and {user}')
            # Slicing the dataframe for the current user and current sensor_comb
            curr_user_df = curr_comb_df.loc[curr_comb_df['user'] == user]
            zero_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'zero-effort']
            dict_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'dict-effort']

            # Getting the row which has max sfars...# the following shall give one or more rows, at least one
            ze_best = zero_effort_df.sort_values(by='far', ascending=False).iloc[0,:]
            de_best = dict_effort_df.sort_values(by='far', ascending=False).iloc[0,:]
    #
            ZeroEffort.loc[zcounter] = [ze_best.user, sensor_dict[sensor_comb], classifiers[cls], ze_best.far, ze_best.frr, ze_best.hter]
            zcounter = zcounter+1

            DictionaryEffort.loc[dcounter] = [de_best.user, sensor_dict[sensor_comb], classifiers[cls], de_best.far, de_best.frr, de_best.hter]
            dcounter = dcounter+1
            if user in DefaultParam.HIGHEFFORT_USER_LIST:
                high_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'high-effort']
                he_best = high_effort_df.sort_values(by='far', ascending=False).iloc[0, :]
                HighEffort.loc[hcounter] = [he_best.user, sensor_dict[sensor_comb], classifiers[cls], he_best.far, he_best.frr, he_best.hter]
                hcounter = hcounter+1


print("************Classifierwise and scenariowise stats***************")
print(ZeroEffort.groupby('classifier').mean().sort_values(by="hter"))
print(DictionaryEffort.groupby('classifier').mean().sort_values(by="hter"))
print(HighEffort.groupby('classifier').mean().sort_values(by="hter"))

print("************Sensorwise and scenariowise stats***************")
print(ZeroEffort.groupby('sensors').mean().sort_values(by="hter"))
print(DictionaryEffort.groupby('sensors').mean().sort_values(by="hter"))
print(HighEffort.groupby('sensors').mean().sort_values(by="hter"))

print("************Userwise and scenariowise stats***************")
print(ZeroEffort.groupby('user').mean().sort_values(by="hter"))
print(DictionaryEffort.groupby('user').mean().sort_values(by="hter"))
print(HighEffort.groupby('user').mean().sort_values(by="hter"))
# ########################################################################################################################
# sns.set()
# # https://seaborn.pydata.org/tutorial/color_palettes.html
# sns.set_palette(sns.diverging_palette(255, 133, l=60, n=7, center="dark"))
#
# ZeroEffortMelted = pd.melt(ZeroEffort, id_vars=['user','sensors'], value_name='error', var_name="errortype")
# plt.subplots(figsize=(10, 3))
# # coolwarm is a good pallet option
# p1 = sns.barplot(x='user', y='far', hue= 'classifier', data=ZeroEffort, order=user_list, edgecolor = 'w')
# plt.grid(linestyle='-', linewidth=0.5, axis='both')
# p1.set_xticklabels(user_list, rotation=90)
# p1.set(ylim=(0, 1), yticks=[x for x in np.linspace(start=0.0, stop=1.0, num=11)])
# # plt.legend(bbox_to_anchor=(1.005, 1.005), ncol=5, loc='best', fontsize=10)
# plt.legend(bbox_to_anchor=(0.5, 0.5, 0.5, 0.5), ncol=1,fontsize=10)
#
# # plt.title('k-Nearest Neighbors Under Zero-effort Attack')
# plt.tight_layout()
#
# ########################################################################################################################
# plt.subplots(figsize=(10, 3))
# # coolwarm is a good pallet option
# p1 = sns.barplot(x='user', y='far', hue= 'classifier', data=DictionaryEffort, order=user_list, edgecolor = 'w')
# plt.grid(linestyle='-', linewidth=0.5, axis='both')
# p1.set_xticklabels(user_list, rotation=90)
# p1.set(ylim=(0, 1), yticks=[x for x in np.linspace(start=0.0, stop=1.0, num=11)])
# # plt.legend(bbox_to_anchor=(1.005, 1.005), ncol=1, loc='best', fontsize=10)
# plt.legend(bbox_to_anchor=(0.31, 0.5, 0.3, 0.5), ncol=1, fontsize=10)
# # plt.title('k-Nearest Neighbors Under Dictionary-effort Attack')
# plt.tight_layout()
#
# ########################################################################################################################
# # HighEffortMelted = pd.melt(HighEffort, id_vars=['user','sensors'], value_name='error', var_name="errortype")
# plt.subplots(figsize=(10, 3))
# # coolwarm is a good pallet option
# p1 = sns.barplot(x='user', y='far', hue= 'classifier', data=HighEffort, order=DefaultParam.HIGHEFFORT_USER_LIST, edgecolor = 'w')
# plt.grid(linestyle='-', linewidth=0.5, axis='both')
# p1.set_xticklabels(DefaultParam.HIGHEFFORT_USER_LIST, rotation=90)
# p1.set(ylim=(0, 1), yticks=[x for x in np.linspace(start=0.0, stop=1.0, num=11)])
# # plt.legend(bbox_to_anchor=(1.005, 1.005), ncol=5, loc='best', fontsize=10)
# plt.legend(bbox_to_anchor=(0.3, 0.5, 0.3, 0.5), ncol=1,fontsize=10)
# # plt.legend(loc='upper center', bbox_to_anchor=(0.2, 0.8, 0.5, 0.5), ncol=5,fontsize=10)
#
# # plt.title('k-Nearest Neighbors Under High-effort Attack')
# plt.tight_layout()
#
# plt.show()
