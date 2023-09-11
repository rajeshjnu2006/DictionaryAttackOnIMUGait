########################################################################################################################
# OLD-effort attack | for all users from the rest of the users
# Dictionary-effort attack | for all users from the 178 samples produced on treadmill
# High-effort attack | for first 18 users who were specifically targeted by one imitator
#                    | Model for the first 18 users shall be trained using the imp data from all possible imp
#                    | and tested only by using the data collected using the feedback model
########################################################################################################################

# !/usr/bin/env python -W ignore::DeprecationWarning
from sklearn.neural_network import MLPClassifier


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import numpy as np
import pandas as pd
from sklearn import linear_model
import os
import sys

# adding the path, else import wouldn't work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from Preprocess import Prep3
from Preprocess import Utilities
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from GlobalParameters import DefaultParam
from FeatureSelection import Supervised

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
########################################################################################################################
gen_user_list = os.listdir(DefaultParam.FEATUREFILE_PATH)
ze_imp_user_list = os.listdir(DefaultParam.FEATUREFILE_PATH)
# making sure the users executed in order so its easy to keep track..
gen_user_list = ['User' + str(id) for id in range(1, len(gen_user_list) + 1) if
                 'User' + str(id) not in DefaultParam.BAD_USERS]
ze_imp_user_list = ['User' + str(id) for id in range(1, len(ze_imp_user_list) + 1) if
                    'User' + str(id) not in DefaultParam.BAD_USERS]

########################################################################################################################
# Sensor configuration
sensor_configurations = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
print(f'#__Running for the following sensor configurations:{sensor_configurations}')
# #####################################################################################################################
# ## Dataframe for storing the results The reference hter is to which it has to be compared, e.g. for zero-effort it
# would be training hter, while for the rest of the scenarios, it would be the zero-effort hter as baseline
ulptable = pd.DataFrame(columns=['user', 'attacktype', 'attacker', 'sensors', 'nfeatures', 'hlayers', 'activation',
                                 'solver', 'alpha', 'learning_rate', 'far', 'frr', 'hter','refhter'])
row_counter = 0
########################################################################################################################
# create a folder to store the results
result_location = os.path.join(os.getcwd(), os.path.basename(__file__)[:-3] + 'Results')
if os.path.exists(result_location):
    # raise FileExistsError("#__The result directory already exists.. delete manually")
    print('removing and creating a new folder')
else:
    os.mkdir(result_location)
    print(f'#__Created a new directory to save results at {result_location}')
########################################################################################################################
if DefaultParam.CLASS_BALANCING:
    print('#__Applying SMOTE for CLASS BALANCING')
# DefaultParam.FEATURE_DOMAIN_LIST = ['TimeFeatures','FrequencyFeatures', 'ITheoryFeatures']
if DefaultParam.FEATURE_DOMAIN_LIST:
    print(f'Using features from: {DefaultParam.FEATURE_DOMAIN_LIST}')
########################################################################################################################
# Setting up the feature selector
# fsselect_method = Unsupervised.ft_using_pca
fsselect_method = Supervised.fs_using_MI
list_num_features = [int(item) for item in np.linspace(start=30, stop=30, num=1)]
########################################################################################################################

print('#__Total users:', len(gen_user_list))
print(f'Genuine user list: {gen_user_list}')
print(f'Zero-effort impostor user list: {ze_imp_user_list}')

########################################################################################################################
# Running the exp for all users, in general. The high-effort though shall run only for first 18 users
# gen_user_list = ['User1', 'User10', 'User12', 'User14']
for user in gen_user_list:
    # sensor_configurations = [('LAcc',), ('Gyr',), ('Mag',), ('RVec',)]
    ####################################################################################################################
    for curr_comb in sensor_configurations:
        # we need to record the biggest damage for each user for each sensor combination
        max_dfar = -1  # the max damage using dict attack
        max_hfar = -1  # the max damage using high effort attack
        ################################################################################################################
        for num_features in list_num_features:
            comb_selected_sens_gtr, comb_selected_sens_gts, comb_selected_sens_itr, comb_selected_sens_its, normalizers, \
            feature_selectors = Prep3.get_feature_matrices(curr_comb, ze_imp_user_list, user, fsselect_method,
                                                           num_features)
            ############################################################################################################
            ###########Training the model and finding the best parameter using 10 fold cross validation
            ############################################################################################################
            training_data = np.vstack((comb_selected_sens_gtr, comb_selected_sens_itr))
            training_labels = np.concatenate(
                (Prep3.get_labels(comb_selected_sens_gtr, 1), Prep3.get_labels(comb_selected_sens_itr, -1)))
            # parameter preparation
            # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
            # use the random grid to search for best hyperparameters
            # first create the base model to tune
            scorerHTER = make_scorer(Utilities.scoringHTER, greater_is_better=False)
            # scorerFAR = make_scorer(Utilities.scoringFAR, greater_is_better=False)
            # scorerFRR = make_scorer(Utilities.scoringFRR, greater_is_better=False)
            hlsize = []

            for flayer in range(100, 301, 100):
                # hlsize.append((flayer,))
                for slayer in range(50, 101, 50):
                    hlsize.append((flayer, slayer))
            # print(hlsize)
            # solver = ['adam', 'lbfgs']
            solver = ['adam']
            # activation = ['relu','tanh']
            activation = ['relu']
            # Read: "https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html"
            # Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by
            # constraining the size of the weights.Increasing alpha may fix high variance (a sign of overfitting) by
            # encouraging smaller weights, resulting in a decision boundary plot that appears with lesser
            # curvatures.Similarly, decreasing alpha may fix high bias (a sign of underfitting) by encouraging larger
            # weights, potentially resulting in a more complicated decision boundary.
            # please look at this while tunning alpha -- the regularization param https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
            alpha = [0.1, 0.2, 0.4, 0.8, 1.6]
            # default is 0.0001, # Experiment with this three values.. higher the alpha, simpler the boundary
            learning_rate = ['adaptive']
            # Create the random grid
            param_grid = {'hidden_layer_sizes': hlsize,
                          'activation': activation,
                          'solver': solver,
                          'alpha': alpha,
                          'learning_rate': learning_rate}
            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            CUAuthModel = MLPClassifier(max_iter=1000, random_state=DefaultParam.RANDOM_SEED, early_stopping=True)
            SearchTheBestParam = GridSearchCV(estimator=CUAuthModel, param_grid=param_grid, cv=5, scoring=scorerHTER)
            SearchTheBestParam.fit(training_data, training_labels)

            hlayers = SearchTheBestParam.best_params_['hidden_layer_sizes']
            activation = SearchTheBestParam.best_params_['activation']
            solver = SearchTheBestParam.best_params_['solver']
            alpha = SearchTheBestParam.best_params_['alpha']
            learning_rate = SearchTheBestParam.best_params_['learning_rate']

            ############################################################################################################
            ########### Validation
            ############################################################################################################
            # Retraining the model again and testing under different attack scenarios
            FinalModel = MLPClassifier(max_iter=1000, hidden_layer_sizes=hlayers, activation=activation,
                                       random_state=DefaultParam.RANDOM_SEED, solver=solver, alpha=alpha,
                                       learning_rate=learning_rate)
            FinalModel.fit(training_data, training_labels)

            pred_gen_lables_tr = FinalModel.predict(comb_selected_sens_gtr)
            pred_imp_lables_tr = FinalModel.predict(comb_selected_sens_itr)
            pred_labels_tr = np.concatenate((pred_gen_lables_tr, pred_imp_lables_tr))
            actual_labels_tr = np.concatenate( (Prep3.get_labels(comb_selected_sens_gtr, 1), Prep3.get_labels(comb_selected_sens_itr, -1)))
            tr_tn, tr_fp, tr_fn, tr_tp = confusion_matrix(actual_labels_tr, pred_labels_tr).ravel()
            tr_far = tr_fp / (tr_fp + tr_tn)
            tr_frr = tr_fn / (tr_fn + tr_tp)
            tr_hter = (tr_far + tr_frr) / 2

            ############################################################################################################
            ########### Zero effort attack scenario
            ############################################################################################################
            pred_gen_lables_zts = FinalModel.predict(comb_selected_sens_gts)
            pred_imp_lables_zts = FinalModel.predict(comb_selected_sens_its)
            pred_labels_zts = np.concatenate((pred_gen_lables_zts, pred_imp_lables_zts))
            actual_labels_zts = np.concatenate(
                (Prep3.get_labels(comb_selected_sens_gts, 1), Prep3.get_labels(comb_selected_sens_its, -1)))

            # computing the error rates for the current predictions
            tsz_tn, tsz_fp, tsz_fn, tsz_tp = confusion_matrix(actual_labels_zts, pred_labels_zts).ravel()
            tsz_far = tsz_fp / (tsz_fp + tsz_tn)
            tsz_frr = tsz_fn / (tsz_fn + tsz_tp)
            tsz_hter = (tsz_far + tsz_frr) / 2
            row_counter = row_counter + 1
            ulptable.loc[row_counter] = [user, "zero-effort", "rest", curr_comb, num_features, hlayers,
                                         activation, solver, alpha, learning_rate,
                                         tsz_far, tsz_frr, tsz_hter, tr_hter]

            ############################################################################################################
            ########### Dictionary-effort attack scenario
            ############################################################################################################
            dict_attacker_list = os.listdir(DefaultParam.DICT_FEATUREFILE_PATH)
            dict_attacker_list = sorted(dict_attacker_list)
            # dict_attacker_list = ['I1_slength_short']
            thebest_dict_attacker = "---"
            for dict_attacker in dict_attacker_list:
                # normalized, selected, attack data
                dict_attack_data = Prep3.get_feature_matrix_for_dict_attack(curr_comb, dict_attacker, feature_selectors,
                                                                            normalizers)
                pred_gen_lables_dts = FinalModel.predict(comb_selected_sens_gts)
                pred_imp_lables_dts = FinalModel.predict(dict_attack_data)  ## Attack
                pred_labels_dts = np.concatenate((pred_gen_lables_dts, pred_imp_lables_dts))

                actual_labels_dts = np.concatenate(
                    (Prep3.get_labels(comb_selected_sens_gts, 1), Prep3.get_labels(dict_attack_data, -1)))
                tsd_tn, tsd_fp, tsd_fn, tsd_tp = confusion_matrix(actual_labels_dts, pred_labels_dts).ravel()
                tsd_far = tsd_fp / (tsd_fp + tsd_tn)
                tsd_frr = tsd_fn / (tsd_fn + tsd_tp)
                tsd_hter = (tsd_far + tsd_frr) / 2
                if tsd_frr != tsz_frr:
                    raise ValueError("False reject rates should be the same...!")
                # Keeping track of the maximum damage
                if tsd_far > max_dfar:
                    max_dfar = tsd_far
                    thebest_dict_attacker = dict_attacker
                # updating the giant perf table
                row_counter = row_counter + 1
                ulptable.loc[row_counter] = [user, "dict-effort", dict_attacker, curr_comb, num_features, hlayers,
                                             activation, solver, alpha, learning_rate,tsd_far, tsd_frr, tsd_hter, tsz_hter]
            ############################################################################################################
            ########### High-effort attack scenario
            ############################################################################################################
            attempt_list = ['Attempt1', 'Attempt2', 'Attempt3']
            the_best_attempt = "---"
            max_hfar = -1
            high_attacker = user  # The attack data is saved by user names
            if user in DefaultParam.HIGHEFFORT_USER_LIST:  # Running only for the first 18 users
                for attempt in attempt_list:
                    # normalized, selected, attack data
                    high_attack_data = Prep3.get_feature_matrix_for_high_attack(curr_comb, high_attacker, attempt,
                                                                                feature_selectors,
                                                                                normalizers)
                    pred_gen_lables_hts = FinalModel.predict(comb_selected_sens_gts)
                    pred_imp_lables_hts = FinalModel.predict(high_attack_data)
                    pred_labels_hts = np.concatenate((pred_gen_lables_hts, pred_imp_lables_hts))

                    actual_labels_hts = np.concatenate(
                        (Prep3.get_labels(comb_selected_sens_gts, 1), Prep3.get_labels(high_attack_data, -1)))
                    tsh_tn, tsh_fp, tsh_fn, tsh_tp = confusion_matrix(actual_labels_hts, pred_labels_hts).ravel()
                    tsh_far = tsh_fp / (tsh_fp + tsh_tn)
                    tsh_frr = tsh_fn / (tsh_fn + tsh_tp)
                    tsh_hter = (tsh_far + tsh_frr) / 2
                    if tsh_frr != tsz_frr:
                        raise ValueError("False reject rates should be the same...!")
                    # Keeping track of the maximum damage
                    if tsh_far > max_hfar:
                        max_hfar = tsh_far
                        the_best_attempt = attempt
                    # updating the giant perf table
                    row_counter = row_counter + 1
                    ulptable.loc[row_counter] = [user, "high-effort", attempt, curr_comb, num_features, hlayers,
                                                 activation, solver, alpha, learning_rate,
                                                 tsh_far, tsh_frr, tsh_hter, tsz_hter]

            print('params:', hlayers, activation, solver, alpha, learning_rate)
            print(f'{user} | {thebest_dict_attacker} | {the_best_attempt}| {curr_comb} | tr_hter:{tr_hter:.3f} | '
                  f'tsz_frr:{tsz_frr:.3f}, tsz_far:{tsz_far:.3f}, tsz_hter:{tsz_hter:.3f} | max dfar: {max_dfar:.3f} '
                  f'| max_hfar:{max_hfar:.3f}')
    cuser_df = ulptable[ulptable.user == user]
    cuser_df.to_csv(os.path.join(result_location, user + '.csv'))

# Writing the full results
ulptable.to_csv(os.path.join(result_location, os.path.basename(__file__)[:-3] + 'combined.csv'))
print(ulptable.to_string())
