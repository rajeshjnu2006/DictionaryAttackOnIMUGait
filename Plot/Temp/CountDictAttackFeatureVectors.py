########################################################################################################################
########### Dictionary-effort attack scenario
########################################################################################################################
import os
import pandas as pd
import numpy as np
from GlobalParameters import DefaultParam
from Preprocess import Utilities, Prep3
folder_path = "D:\ThesisExperiments\Storage\FeatureFilesDictionary"
dict_attacker_list = os.listdir(folder_path)
dict_attacker_list = sorted(dict_attacker_list)
number_of_fv = []
for dict_attacker in dict_attacker_list:
    df = pd.read_csv(os.path.join(folder_path,dict_attacker,r"TimeFeatures\feat_clean_LAcc.csv"))
    number_of_fv.append(df.shape[0])

number_of_fv.sort()
print('Mean:',np.mean(number_of_fv))
print('Std:',np.std(number_of_fv))
print('Min:',np.min(number_of_fv))
print('Max:',np.max(number_of_fv))

print('Median:',np.median(number_of_fv))

print(number_of_fv)