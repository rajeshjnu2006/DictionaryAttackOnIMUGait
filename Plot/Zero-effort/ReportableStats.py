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
classifiers = {'../kNNZDHcombined.csv':'kNN', '../LRegZDHcombined.csv':'LReg', '../MLPZDHcombined.csv':'MLP','../RanForZDHcombined.csv':'RanFor','../SVMZDHcombined.csv':'SVM'}
for cls in classifiers:
    results_df = pd.read_csv(cls, dtype={'sensors': str}, index_col=0)
    for sensor_comb in sensor_dict:
        curr_comb_df = results_df.loc[results_df['sensors'] == str(sensor_comb)]  # slicing the dataframe for the current
        # sensor comb
        # Traversing all the users for the current sensor
        # user_list = ['User1', 'User2']
        for user in user_list:
            # print(f'running for {sensor_comb} and {user}')
            # Slicing the dataframe for the current user and current sensor_comb
            curr_user_df = curr_comb_df.loc[curr_comb_df['user'] == user]
            zero_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'zero-effort']
            # Getting the row which has max sfars...# the following shall give one or more rows, at least one
            ze_best = zero_effort_df.sort_values(by='far', ascending=False).iloc[0,:]  # No need to do this, just following the pattern of sorting and picking the best
            ZeroEffort.loc[zcounter] = [sensor_dict[sensor_comb], classifiers[cls], ze_best.user, ze_best.far, ze_best.frr, ze_best.hter,
                                        ze_best.refhter]
            zcounter = zcounter + 1

########################################################################################################################
sns.set()
# https://seaborn.pydata.org/tutorial/color_palettes.html
# sns.set_palette(sns.diverging_palette(255, 133, l=60, n=7, center="dark"))
sns.set_palette('coolwarm')

ZeroEffortMelted = pd.melt(ZeroEffort, id_vars=['sensors', 'user', 'zero_far', 'zero_frr', 'zero_hter', 'tr_hter'], value_name='classifier', var_name="clstype")

print("************Classifierwise and scenariowise stats***************")
print(ZeroEffort.groupby('classifier').mean().sort_values(by="zero_hter"))
print("************Sensorwise and scenariowise stats***************")
print(ZeroEffort.groupby('sensors').mean().sort_values(by="zero_hter"))

print("************Userwise and scenariowise stats***************")
print(ZeroEffort.groupby('user').mean().sort_values(by="zero_hter"))