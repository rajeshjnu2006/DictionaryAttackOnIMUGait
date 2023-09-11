########################################################################################################################
########### Dictionary-effort attack scenario
########################################################################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##############################################################
print("******ZERO**********")
folder_path = "D:\ThesisExperiments\Storage\FeatureFilesGenuine"
user_list_dirs = os.listdir(folder_path)
user_list = []
for id in range(1,len(user_list_dirs)+1):
    user_list.append('User'+str(id))

number_of_fv_train = []
number_of_fv_test = []
for user in user_list:
    df_train = pd.read_csv(os.path.join(folder_path,user,r"Training\TimeFeatures\feat_clean_LAcc.csv"))
    df_test = pd.read_csv(os.path.join(folder_path,user,r"Testing\TimeFeatures\feat_clean_LAcc.csv"))
    number_of_fv_train.append(df_train.shape[0])
    number_of_fv_test.append(df_test.shape[0])

plt.figure('Zero-effort-fvstats', figsize=(10,3))
Gen_details= pd.DataFrame(list(zip(user_list,number_of_fv_train,number_of_fv_test)), columns=['Users','Training','Testing'])
Gen_details= pd.melt(Gen_details, id_vars =['Users'], value_vars =['Training','Testing'], var_name ='Sessions', value_name ='# feature vectors')
ax = sns.barplot(x="Users", y="# feature vectors", hue='Sessions', data=Gen_details)
plt.xticks(np.linspace(0,len(user_list),len(user_list), endpoint=False), user_list, rotation='vertical')
plt.yticks([int(i) for i in np.linspace(0,100,11, endpoint=True)], [int(j) for j in np.linspace(0,100,11, endpoint=True)])
plt.subplots_adjust(left=0.09, bottom=0.17, right=.98, top=0.80, wspace=0.20, hspace=0.30)
plt.tight_layout()
ax.legend(ncol=1,loc='upper left')
plt.grid(True)
# ax.grid(which='major', axis='y', linestyle='--')
# plt.show()

#
##############################################################
print("******HIGH**********")
folder_path = "D:\ThesisExperiments\Storage\FeatureFilesHigh"
user_list_dirs = os.listdir(folder_path)
user_list = []
for id in range(1,len(user_list_dirs)+1):
    user_list.append('User'+str(id))
number_of_fv_high = []
df_attempt1_list = []
df_attempt2_list = []
df_attempt3_list = []
for user in user_list:
    df_attempt1 = pd.read_csv(os.path.join(folder_path,user,r"Attempt1\TimeFeatures\feat_clean_LAcc.csv"))
    df_attempt2 = pd.read_csv(os.path.join(folder_path,user,r"Attempt2\TimeFeatures\feat_clean_LAcc.csv"))
    df_attempt3 = pd.read_csv(os.path.join(folder_path,user,r"Attempt3\TimeFeatures\feat_clean_LAcc.csv"))
    number_of_fv_high.append(int((df_attempt1.shape[0]+df_attempt2.shape[0]+df_attempt3.shape[0])/3))
    df_attempt1_list.append(df_attempt1.shape[0])
    df_attempt2_list.append(df_attempt2.shape[0])
    df_attempt3_list.append(df_attempt3.shape[0])

plt.figure('High-effort-fvstats', figsize=(10,3))
High_details= pd.DataFrame(list(zip(user_list,df_attempt1_list, df_attempt2_list, df_attempt3_list)), columns=['Users','Attempt1','Attempt2','Attempt3'])
High_details= pd.melt(High_details, id_vars =['Users'], value_vars =['Attempt1','Attempt2','Attempt3'], var_name ='Sessions', value_name ='# feature vectors')
ax = sns.barplot(x="Users", y="# feature vectors", hue='Sessions', data=High_details)
plt.xticks(np.linspace(0,len(user_list),len(user_list), endpoint=False), user_list, rotation='vertical')
plt.yticks([int(i) for i in np.linspace(0,100,11, endpoint=True)], [int(j) for j in np.linspace(0,100,11, endpoint=True)])
plt.subplots_adjust(left=0.09, bottom=0.17, right=.98, top=0.80, wspace=0.20, hspace=0.30)
ax.legend(ncol=1,loc='upper right')
plt.tight_layout()
plt.grid(True)
# plt.show()


##############################################################
print("******DICT**********")

folder_path = "D:\ThesisExperiments\Storage\FeatureFilesDictionary"
dict_attacker_list = os.listdir(folder_path)
dict_attacker_list = sorted(dict_attacker_list)
number_of_fv_dict = []
for dict_attacker in dict_attacker_list:
    df = pd.read_csv(os.path.join(folder_path,dict_attacker,r"TimeFeatures\feat_clean_LAcc.csv"))
    number_of_fv_dict.append(df.shape[0])
cm = sns.cubehelix_palette(6)
plt.figure('Dict-effort-fvstats', figsize=(10,3))
High_details = pd.DataFrame(list(zip(dict_attacker_list,number_of_fv_dict)), columns=['Unique Patterns','# feature vectors'])
# High_details = pd.melt(High_details, id_vars =['Users'], value_vars =['Attack'], var_name ='Sessions', value_name ='# feature vectors')
print(High_details)
ax = sns.barplot(x="Unique Patterns", y="# feature vectors", data=High_details)#, palette="Set3")
# plt.xticks(np.linspace(0,len(dict_attacker_list),len(dict_attacker_list), endpoint=False), [str(i) for i in range(1,len(dict_attacker_list)+1)], rotation='vertical')

num_ticks = (len(dict_attacker_list)+2)/4
plt.xticks(np.linspace(0,len(dict_attacker_list), num=num_ticks, endpoint=True), [str(int(i)) for i in np.linspace(1,len(dict_attacker_list)+1, num=num_ticks, endpoint=True)], rotation='vertical')
# plt.xticks([])#, dict_attacker_list, rotation='vertical')
plt.yticks([int(i) for i in np.linspace(0,100,11, endpoint=True)], [int(j) for j in np.linspace(0,100,11, endpoint=True)])

plt.subplots_adjust(left=0.09, bottom=0.17, right=.98, top=0.80, wspace=0.20, hspace=0.30)
# ax.legend(ncol=1, loc='upper right')
plt.tight_layout()
plt.grid(True)
plt.show()
