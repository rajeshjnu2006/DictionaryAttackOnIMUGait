#!/usr/bin/env python -W ignore::DeprecationWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import warnings
import numpy as np
import pandas as pd
from sklearn import linear_model
import os
import sys
# adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

sys.path.insert(0, os.path.dirname(os.getcwd()))
from Preprocess import Prep
from Preprocess import Utilities
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from GlobalParameters import DefaultParam
from FeatureSelection import Supervised

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
#___________________________________________________________________________________________________________________#
genuine_user_list = os.listdir(DefaultParam.FEATUREFILE_PATH)
imposter_list = os.listdir(DefaultParam.FEATUREFILE_PATH)
# making sure the users executed in order so its easy to keep track..
ordered_genuine_users = ['User'+str(id) for id in range(1, len(genuine_user_list)+1)]
# removing the bad users -- a provision for future if we need to take some bad users out
if DefaultParam.REMOVE_BAD_USERS:
    for item in DefaultParam.BAD_USERS:
        ordered_genuine_users.remove(item)

print('#_________________Total users:', len(ordered_genuine_users))
print(f'#_________________Running for total {len(ordered_genuine_users)} genuine users, the list includes {ordered_genuine_users}')
#___________________________________________________________________________________________________________________#
# setting up sensors and its combinations
# data preprocessor setting and flags
sense_comb_list = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
print(f'#_________________Running for the following sensor configurations:{sense_comb_list}')
#___________________________________________________________________________________________________________________#
### setting up the result dataframe##############
# creating dataframes for storing performances
# creating frame to storing errors
ulptable = pd.DataFrame(columns=['User', 'Sensors', 'Num_features', 'CVal', 'Solver', 'Penalty','Training_score','FAR', 'FRR', 'HTER'])
row_counter = 0
# create a folder to store the results
result_location = os.path.join(os.getcwd(), os.path.basename(__file__)[:-3] + 'Results')
if os.path.exists(result_location):
    # raise FileExistsError("#_________________The result directory already exists.. delete manually")
    print('removing and creating a new folder')
else:
    os.mkdir(result_location)
    print(f'#_________________Created a new directory to save results at {result_location}')
# ___________________________________________________________________________________________________________________#
if DefaultParam.CLASS_BALANCING:
    print('#_________________Applying SMOTE for CLASS BALANCING')
# ___________________________________________________________________________________________________________________#
if DefaultParam.FEATURE_DOMAIN_LIST:
    print(f'Using features from: {DefaultParam.FEATURE_DOMAIN_LIST}')

# using this first time and feeling very good about it .. function as first class object
# fsselect = Unsupervised.ft_using_pca
fsselect = Supervised.fs_using_MI
fsthresholds = np.linspace(start=30, stop=60, num=20)  # The lesser the better USE THIS EVERYWHERE
fsthresholds = [int(item) for item in fsthresholds]


ordered_genuine_users = ['User3', 'User4', 'User5', 'User6', 'User7']
# ___________________________________________________________#
for user in ordered_genuine_users:
    print(f'#_________________{user}_____________________________#')
    # sense_comb_list = [('LAcc',), ('Gyr',), ('Mag',), ('RVec',)]
    for curr_comb in sense_comb_list:
        # Deleting the variables if it already exists
        if 'com_gen_tr_data' in locals(): # If one exists all would do
            del com_gen_tr_data
            del com_gen_ts_data
            del com_imp_tr_data
            del com_imp_ts_data
        # print(f'curr_sensor_comb{curr_comb}')
        # overriding the num_imp sample parameter
        # preparing data based on what domain of features we have to include from what sensors
        #_________________________________________________________________________________________________________#
        for sensor in curr_comb:
            for fdomain in DefaultParam.FEATURE_DOMAIN_LIST:
                if 'com_gen_tr_data' not in locals():
                    com_gen_tr_data, com_gen_ts_data, com_imp_tr_data, com_imp_ts_data = Prep.get_data(DefaultParam.FEATUREFILE_PATH, fdomain,
                                                                                                       sensor, imposter_list, user, DefaultParam.NUM_SAMPLE_IMP)
                else:
                    gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data = Prep.get_data(DefaultParam.FEATUREFILE_PATH, fdomain,
                                                                                       sensor, imposter_list, user, DefaultParam.NUM_SAMPLE_IMP)
                    # damage control, just in case due to the error in sampling rate calculation the num of
                    # rows varies across sensors
                    min_gen_tr_samples = min(com_gen_tr_data.shape[0], gen_tr_data.shape[0])
                    min_gen_ts_samples = min(com_gen_ts_data.shape[0], gen_ts_data.shape[0])
                    min_imp_tr_samples = min(com_imp_tr_data.shape[0], imp_tr_data.shape[0])
                    min_imp_ts_samples = min(com_imp_ts_data.shape[0], imp_ts_data.shape[0])
                    # print('min(com_gen_ts_data.shape[0], gen_ts_data.shape[0]), min_gen_ts_samples:',
                    # com_gen_ts_data.shape[0], gen_ts_data.shape[0],min_gen_ts_samples) ignore_index = True -===>
                    # removes all the named reference -- strange join_axes=[com_gen_tr_data.index]  preserves th
                    # indices
                    com_gen_tr_data = pd.concat([com_gen_tr_data.iloc[0:min_gen_tr_samples,:], gen_tr_data.iloc[0:min_gen_tr_samples,:]], axis=1)
                    com_gen_ts_data = pd.concat([com_gen_ts_data.iloc[0:min_gen_ts_samples,:], gen_ts_data.iloc[0:min_gen_ts_samples,:]], axis=1)
                    com_imp_tr_data = pd.concat([com_imp_tr_data.iloc[0:min_imp_tr_samples,:], imp_tr_data.iloc[0:min_imp_tr_samples,:]], axis=1)
                    com_imp_ts_data = pd.concat([com_imp_ts_data.iloc[0:min_imp_ts_samples,:], imp_ts_data.iloc[0:min_imp_ts_samples,:]], axis=1)
        #_________________________________________________________________________________________________________#
        feature_names  = com_gen_tr_data.columns
        # Perform feature normalization regardless what algorithms you would apply
        # YOU MUST PERFORM THE NORMALIZATION BEFORE APPLYING THE SMOTE BECAUSE SMOTE USES kNN and IS SENSITITVE TO FEATURE VALUE RANGE
        com_gen_tr_data, com_gen_ts_data, com_imp_tr_data, com_imp_ts_data = Prep.get_normalized_data_using_standard_scaler(
                com_gen_tr_data, com_gen_ts_data, com_imp_tr_data, com_imp_ts_data)
        # _________________________________________________________________________________________________________#


        # Saving feature names before passing
        if DefaultParam.CLASS_BALANCING:
            com_gen_tr_data, com_gen_ts_data, com_imp_tr_data, com_imp_ts_data = Prep.get_balanced_data(com_gen_tr_data, com_gen_ts_data, com_imp_tr_data, com_imp_ts_data)

        # _________________________________________________________________________________________________________#
        # You should use pipeline --- such has been the wait
        # https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html But this is what you
        # would like to show them that the corr or pca affect the results and how
        Training_stats = []
        for fs_threshold in fsthresholds:
            com_gen_tr_data_selected, com_gen_ts_data_selected, com_imp_tr_data_selected, com_imp_ts_data_selected = fsselect(
                com_gen_tr_data, com_gen_ts_data, com_imp_tr_data,
                com_imp_ts_data, fs_threshold)
            # https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
            training_data = np.vstack((com_gen_tr_data_selected, com_imp_tr_data_selected))
            training_labels = np.concatenate((Prep.get_labels(com_gen_tr_data, 1),
                                              Prep.get_labels(com_imp_tr_data, -1)))
            # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
            # Create a dictionary of ParametersRandFor
            # Create the random grid
            # param_grid = [{'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs', 'sag'], 'C': [0.01, 0.1, 1.0], 'penalty': ['l1','l2']}] # This works
            param_grid = [
                {'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs', 'sag'],
                 'C': [0.005, 0.01, 0.02, 0.04, 0.08, 0.16],
                 'penalty': ['l1', 'l2']}]  # Trying to improve

            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            LRegModel = linear_model.LogisticRegression(random_state=DefaultParam.RANDOM_SEED, tol=1e-5)
            # Instantiate the grid search model
            # random search of parameters, using 5 fold cross validation,
            scorerHTER = make_scorer(Utilities.scoringHTER, greater_is_better=False)
            # scorerFAR = make_scorer(Utilities.scoringFAR, greater_is_better=False)
            # scorerFRR = make_scorer(Utilities.scoringFRR, greater_is_better=False)
            scoring_function = 'f1'
            reverse_order = True  ## The higher the better -- this is useful while selecting te best
            LRGSearch = GridSearchCV(LRegModel, param_grid, cv=10, scoring=scoring_function)

            LRGSearch.fit(training_data, training_labels)
            solver = LRGSearch.best_params_['solver']
            cval = LRGSearch.best_params_['C']
            penalty = LRGSearch.best_params_['penalty']
            Training_stats.append([fs_threshold, solver, cval, penalty, LRGSearch.best_score_])

        best_params = Utilities.get_best_params_median_features(Training_stats, reverse_order)
        # To consider in future: there might be many params setting with the best
        best_fstheshold = best_params[0]
        best_solver = best_params[1]
        best_cval = best_params[2]
        best_penalty = best_params[3]
        best_training_score = best_params[4]

        # To retrain, selecting the features again using the best threshold based on the training score
        com_gen_tr_data_selected, com_gen_ts_data_selected, com_imp_tr_data_selected, com_imp_ts_data_selected = fsselect(
            com_gen_tr_data, com_gen_ts_data, com_imp_tr_data,
            com_imp_ts_data, best_fstheshold)

        # Retraining the model using the best ParametersRandFor found from the training dataset
        training_data = np.vstack((com_gen_tr_data_selected, com_imp_tr_data_selected))
        training_labels = np.concatenate(
            (Prep.get_labels(com_gen_tr_data_selected, 1), Prep.get_labels(com_imp_tr_data_selected, -1)))
        FinalModel = linear_model.LogisticRegression(solver=solver, C=cval, penalty=penalty,
                                                     random_state=DefaultParam.RANDOM_SEED, tol=1e-5)
        FinalModel.fit(training_data, training_labels)

        # testing for the Genuine
        pred_genuine_lables = FinalModel.predict(com_gen_ts_data_selected)
        # testing for the impostors
        pred_impostor_lables = FinalModel.predict(com_imp_ts_data_selected)
        pred_labels = np.concatenate((pred_genuine_lables, pred_impostor_lables))
        actual_labels = np.concatenate((Prep.get_labels(com_gen_ts_data_selected, 1), Prep.get_labels(com_imp_ts_data_selected, -1)))
        # computing the error rates for the current predictions
        tn, fp, fn, tp = confusion_matrix(actual_labels, pred_labels).ravel()
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        hter = (far + frr) / 2
        row_counter = row_counter + 1
        num_features = com_gen_ts_data_selected.shape[1]
        # examine the best model
        # https://pyformat.info/
        print(
            f'{user} with sensor_comb:{curr_comb}, num_features: {best_fstheshold}, CVal:{best_cval}, Solver:{best_solver}, Penalty:{best_penalty}, training score:{best_training_score:.4f},far:{far:.3f}, frr:{frr:0.3f} hter:{hter:0.3f}')
        ulptable.loc[row_counter] = [user, curr_comb, best_fstheshold, best_cval, best_solver, best_penalty,
                                     best_training_score, far, frr, hter]
    cuser_df = ulptable[ulptable.User == user]
    cuser_df.to_csv(os.path.join(result_location, user + '.csv'))

# Writing the full results
ulptable.to_csv(os.path.join(result_location, os.path.basename(__file__)[:-3] + 'full_results.csv'))

user_counter = 0
best_perf_table = pd.DataFrame(
    columns=['User', 'Sensors', 'Num_features', 'CVal', 'Solver', 'Penalty', 'Training_score', 'FAR', 'FRR', 'HTER'])

for user in ordered_genuine_users:
    cuser_df = ulptable[ulptable.User == user]
    for curr_comb in sense_comb_list:
        cc_df = cuser_df[cuser_df.Sensors == curr_comb]
        best_perf_table[user_counter] = cc_df[cc_df.HTER == cc_df['HTER'].min()]
        user_counter = user_counter + 1

print(best_perf_table)
