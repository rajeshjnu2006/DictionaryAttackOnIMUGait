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
user_list = ["User"+str(i) for i in range(1,56)]
result_file_path = r'../RanForZDHcombined.csv'
results_df = pd.read_csv(result_file_path, dtype={'sensors': str}, index_col=0)
sense_comb_list = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
sensor_dict = dict(zip(sense_comb_list, Utilities.getallcombinations_sortened(sense_comb_list)))
sensor_name_list = []
for key in sensor_dict:
    # print(f"key:{key} {sensor_dict[key]}")
    sensor_name_list.append(sensor_dict[key])
# creating an ordered user list for sanity
ordered_user_list = []
for i in range(1, len(user_list) + 1):
    ordered_user_list.append('User' + str(i))

########################################################################################################################
ZeroEffort = pd.DataFrame(columns=['user', 'sensors','zero_far', 'zero_frr', 'zero_hter', 'tr_hter'])
DictionaryEffort = pd.DataFrame(columns=['user', 'sensors', 'attacker', 'zero_far', 'dict_far', 'zero_hter', 'dict_hter'])
HighEffort = pd.DataFrame(columns=['user', 'sensors', 'zero_far', 'high_far', 'zero_hter', 'hight_hter'])

# # Overrriding this for individual sensors
# sense_comb_list = [('LAcc',),('Gyr',),('Mag',),('RVec',)]
# sensor_dict = dict(zip(sense_comb_list, ['a','g','m','r']))

########################################################################################################################
zcounter = 0
dcounter = 0
hcounter = 0
for sensor_comb in sensor_dict:
    curr_comb_df = results_df.loc[results_df['sensors'] == str(sensor_comb)]  # slicing the dataframe for the current
    # sensor comb
    # Traversing all the users for the current sensor
    # ordered_user_list = ['User1', 'User2']
    for user in ordered_user_list:
        # print(f'running for {sensor_comb} and {user}')
        # Slicing the dataframe for the current user and current sensor_comb
        curr_user_df = curr_comb_df.loc[curr_comb_df['user'] == user]
        zero_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'zero-effort']
        dict_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'dict-effort']

        # Getting the row which has max sfars...# the following shall give one or more rows, at least one
        ze_best = zero_effort_df.sort_values(by='far', ascending=False).iloc[0,:]
        de_best = dict_effort_df.sort_values(by='far', ascending=False).iloc[0,:]
#
        ZeroEffort.loc[zcounter] = [ze_best.user, sensor_dict[sensor_comb], ze_best.far, ze_best.frr, ze_best.hter, ze_best.refhter]
        zcounter = zcounter+1

        DictionaryEffort.loc[dcounter] = [de_best.user, sensor_dict[sensor_comb], de_best.attacker, ze_best.far, de_best.far, ze_best.hter, de_best.hter]
        dcounter = dcounter+1

        if user in DefaultParam.HIGHEFFORT_USER_LIST:
            high_effort_df = curr_user_df.loc[curr_user_df['attacktype'] == 'high-effort']
            he_best = high_effort_df.sort_values(by='far', ascending=False).iloc[0, :]

            HighEffort.loc[hcounter] = [he_best.user, sensor_dict[sensor_comb], ze_best.far, he_best.far, ze_best.hter, he_best.hter]
            hcounter = hcounter+1
########################################################################################################################
sns.set()
sns.set_palette("Paired")

ZeroEffortMelted = pd.melt(ZeroEffort, id_vars=['user','sensors'], value_name='error', var_name="errortype")
plt.subplots(figsize=(9, 3.5))
# coolwarm is a good pallet option
p1 = sns.barplot(x='sensors', y='error', hue= 'errortype', data=ZeroEffortMelted, order=sensor_name_list,edgecolor = 'w',errwidth=0.6)
plt.grid(linestyle='-', linewidth=0.5, axis='both')
p1.set_xticklabels(sensor_name_list, rotation=90)
p1.set(ylim=(0, 1), yticks=[x for x in np.linspace(start=0.0, stop=1.0, num=11)])
# plt.legend(bbox_to_anchor=(1.005, 1.005), ncol=1)
plt.legend(ncol=4,loc='upper center')

plt.title('Random Forest Under Zero-effort Attack')
plt.tight_layout()

########################################################################################################################
DictionaryEffortMelted = pd.melt(DictionaryEffort, id_vars=['user','sensors', 'attacker'], value_name='error', var_name="errortype")
print(DictionaryEffortMelted)
plt.subplots(figsize=(9, 3.5))
# coolwarm is a good pallet option
p1 = sns.barplot(x='sensors', y='error', hue= 'errortype', data=DictionaryEffortMelted, order=sensor_name_list, edgecolor = 'w',errwidth=0.6)
plt.grid(linestyle='-', linewidth=0.5, axis='both')
p1.set_xticklabels(sensor_name_list, rotation=90)
p1.set(ylim=(0, 1), yticks=[x for x in np.linspace(start=0.0, stop=1.0, num=11)])
plt.legend(bbox_to_anchor=(1.005, 1.005), ncol=1)
plt.legend(ncol=4,loc='upper center')
plt.title('Random Forest Under Dictionary-effort Attack')
plt.tight_layout()

########################################################################################################################
HighEffortMelted = pd.melt(HighEffort, id_vars=['user','sensors'], value_name='error', var_name="errortype")
plt.subplots(figsize=(9, 3.5))
# coolwarm is a good pallet option
p1 = sns.barplot(x='sensors', y='error', hue= 'errortype', data=HighEffortMelted, order=sensor_name_list, edgecolor = 'w',errwidth=0.6)
plt.grid(linestyle='-', linewidth=0.5, axis='both')
p1.set_xticklabels(sensor_name_list, rotation=90)
p1.set(ylim=(0, 1), yticks=[x for x in np.linspace(start=0.0, stop=1.0, num=11)])
# plt.legend(bbox_to_anchor=(1.005, 1.005), ncol=1)
plt.legend(ncol=4,loc='upper center')
plt.title('Random Forest Under High-effort Attack')
plt.tight_layout()

plt.show()
