import os
import sys

import pandas as pd

from Preprocess import Utilities


# Generating a table of top three attackers in each (single and fused) categories
# kNN was the best classifier in the 'a' category while SVM was best in the a+g+r (Base don lowest HTER))
sys.path.insert(0, os.path.dirname(os.getcwd()))
from GlobalParameters import DefaultParam
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

########################################################################################################################
sense_comb_list = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
sensor_dict = dict(zip(sense_comb_list, Utilities.getallcombinations_sortened(sense_comb_list)))
# Individual best
# sense_comb_list = [('LAcc',)]
# sensor_dict = dict(zip(sense_comb_list, [r"'a'"]))
# Comb best
sense_comb_list = [('LAcc', 'Gyr', 'RVec')]
sensor_dict = dict(zip(sense_comb_list, [r"'a+g+r'"]))

sensor_name_list = []
for key in sensor_dict:
    sensor_name_list.append(sensor_dict[key])
########################################################################################################################
# creating an ordered user list for sanity
user_list = []
for i in range(1, 56):
    user_list.append('User' + str(i))

########################################################################################################################
DictionaryEffort = pd.DataFrame(columns=['sensors', 'classifier', 'user', 'attacker', 'zero_far', 'dict_far', 'zero_hter', 'dict_hter'])
########################################################################################################################
dcounter = 0
number_of_attackers = 178
classifier_dict = {'../ResultFiles/kNNZDHcombined.csv': 'kNN', '../ResultFiles/LRegZDHcombined.csv': 'LReg', '../ResultFiles/MLPZDHcombined.csv': 'MLP', '../ResultFiles/RanForZDHcombined.csv': 'RanFor', '../ResultFiles/SVMZDHcombined.csv': 'SVM'}
# classifier_dict = {'../ResultFiles/SVMZDHcombined.csv': 'SVM'} ## OVerrriding for the best classifiers for "a"
classifier_dict = {'../ResultFiles/kNNZDHcombined.csv': 'kNN'} ## OVerrriding for the best classifiers for "a+g+r"


cls_name_list = []
for key in classifier_dict:
    cls_name_list.append(classifier_dict[key])

########################################################################################################################
def get_short_name(text):
    if "speed" in text:
        text = text.replace("speed", "SP")
        if "1_4" in text:
            text = text.replace("1_4", "_"+str(int(44.704*1.4)))
        elif "1_6" in text:
            text = text.replace("1_6", "_"+str(int(44.704*1.6)))
        elif "1_8" in text:
            text = text.replace("1_8", "_"+str(int(44.704*1.8)))
        elif "2_0" in text:
            text = text.replace("2_0", "_"+str(int(44.704*2.0)))
        elif "2_2" in text:
            text = text.replace("2_2", "_"+str(int(44.704*2.2)))
        elif "2_4" in text:
            text = text.replace("2_4", "_"+str(int(44.704*2.4)))
        elif "2_6" in text:
            text = text.replace("2_6", "_"+str(int(44.704*2.6)))
        elif "2_8" in text:
            text = text.replace("2_8", "_"+str(int(44.704*2.8)))
        elif "3_0" in text:
            text = text.replace("3_0", "_"+str(int(44.704*3.0)))
        else:
            pass
    elif "slength" in text:
        text = text.replace("slength", "SL")
    elif "swidth" in text:
        text = text.replace("swidth", "SW")
    elif "tlift" in text:
        text = text.replace("tlift", "TL")
    return text
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
            de_best = dict_effort_df.sort_values(by='far', ascending=False).iloc[0:number_of_attackers, :]

            for i in range(0, number_of_attackers):
                # Getting the row which has max sfars...# the following shall give one or more rows, at least one
                de_best_temp = de_best.iloc[i,:]

                DictionaryEffort.loc[dcounter] = [sensor_dict[sensor_comb], classifier_dict[cls], user, get_short_name(de_best_temp.attacker), ze_best.far,
                                              de_best_temp.far, ze_best.hter, de_best_temp.hter]
                dcounter = dcounter + 1

########################################################################################################################
DictionaryEffort['zero_far'] =  DictionaryEffort['zero_far']*100
DictionaryEffort['dict_far'] =  DictionaryEffort['dict_far']*100
DictionaryEffort['zero_hter'] =  DictionaryEffort['zero_hter']*100
DictionaryEffort['dict_hter'] =  DictionaryEffort['dict_hter']*100
TopAttackDict = pd.DataFrame(columns=['User','zfar','top_attackers', 'Average'])

counter= 0
for i, user in enumerate(user_list):
    user_df = DictionaryEffort.loc[DictionaryEffort['user'] == user]
    temp_df  = user_df.iloc[0,:]
    at_list = list(user_df['attacker'])
    dfar_list = list(user_df['dict_far'])
    dfar_list = [int(item) for item in dfar_list]
    l_for_avg = [dfar for dfar in dfar_list if int(dfar) >= int(temp_df.zero_far)]
    att_list = [str(at)+", "+str(dfar) for at, dfar in zip(at_list,dfar_list) if int(dfar) >= int(temp_df.zero_far)] #
    # Removing everything that is less than the zero_far
    if len(att_list)!=0: # Not including the ones with all zeros
        TopAttackDict.loc[counter] = [temp_df.user, int(temp_df.zero_far), att_list, int(np.mean(l_for_avg))]
        counter=counter+1

TopAttackDict = TopAttackDict.set_index('User')
TopAttackDict = TopAttackDict.sort_values(by='Average',ascending=False)
file_name = cls_name_list[0]+"_"+sensor_name_list[0]+'_TopAttackDict.csv'
print(file_name)
TopAttackDict.to_csv(file_name)
