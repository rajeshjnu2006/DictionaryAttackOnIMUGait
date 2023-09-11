#!/usr/bin/env python -W ignore::DeprecationWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
#__________#
genuine_user_list = os.listdir(DefaultParam.FEATUREFILE_PATH)
imposter_user_list = os.listdir(DefaultParam.FEATUREFILE_PATH)
# making sure the users executed in order so its easy to keep track..
ordered_genuine_users = ['User'+str(id) for id in range(1, len(genuine_user_list)+1) if 'User'+str(id) not in DefaultParam.BAD_USERS]
imposter_user_list = ['User' + str(id) for id in range(1, len(imposter_user_list) + 1) if 'User' + str(id) not in DefaultParam.BAD_USERS]

#__________#
# setting up sensors and its combinations
# data preprocessor setting and flags
comb_fdomain_list = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
print(f'#__Running for the following sensor configurations:{comb_fdomain_list}')
#__________#
### setting up the result dataframe##############
# creating dataframes for storing performances
# creating frame to storing errors
ulptable_zts = pd.DataFrame(columns=['User', 'Sensors', 'nn', 'dist', 'test_far', 'test_frr', 'test_hter', 'train_hter'])
row_counter = 0
# create a folder to store the results
result_location = os.path.join(os.getcwd(), os.path.basename(__file__)[:-3] + 'Results')
if os.path.exists(result_location):
    # raise FileExistsError("#__The result directory already exists.. delete manually")
    print('removing and creating a new folder')
else:
    os.mkdir(result_location)
    print(f'#__Created a new directory to save results at {result_location}')
# __________#
if DefaultParam.CLASS_BALANCING:
    print('#__Applying SMOTE for CLASS BALANCING')
# __________#
# DefaultParam.FEATURE_DOMAIN_LIST = ['TimeFeatures','FrequencyFeatures', 'ITheoryFeatures']
if DefaultParam.FEATURE_DOMAIN_LIST:
    print(f'Using features from: {DefaultParam.FEATURE_DOMAIN_LIST}')
# using this first time and feeling very good about it .. function as first class object
# fsselect = Unsupervised.ft_using_pca
fsselect = Supervised.fs_using_MI
# taking 30, 40, or 50 features from each participating sensor
list_num_features = [int(item) for item in np.linspace(start=30, stop=30, num=1)]

# Following for the testing purposes

print('#__Total users:', len(ordered_genuine_users))
print(f'#__Running for total {len(ordered_genuine_users)} genuine users, the list includes {ordered_genuine_users}')
print(f'#__Running for total {len(imposter_user_list)} impostor users, the list includes {imposter_user_list}')


for user in ordered_genuine_users:
    # comb_fdomain_list = [('LAcc',), ('Gyr',), ('Mag',), ('RVec',)]
    for curr_comb in comb_fdomain_list:
        if 'gtr_fm' in locals():  # If one exists all would do
            del comb_sens_gtr
            del comb_sens_gts
            del comb_sens_itr
            del comb_sens_its
        separating_indices = [0]
        for sensor in curr_comb:
            if 'comb_fdomain_gtr' in locals():  # If one exists all would do
                del comb_fdomain_gtr
                del comb_fdomain_gts
                del comb_fdomain_itr
                del comb_fdomain_its
            # Keeping track of the indices that separate sensors in the feature matrices: practically it should be the
            # multiple of total number of features per sensor
            for fdomain in DefaultParam.FEATURE_DOMAIN_LIST:
                # Checking for the first call
                if 'comb_fdomain_gtr' not in locals():
                    # print(f'User:{user}, Curr_comb:{curr_comb}, Sensor:{sensor}, FDomain:{fdomain}')
                    comb_fdomain_gtr, comb_fdomain_gts, comb_fdomain_itr, comb_fdomain_its = Prep.get_data(DefaultParam.FEATUREFILE_PATH, fdomain, sensor, imposter_user_list, user, DefaultParam.NUM_SAMPLE_IMP)
                else:
                    # print(f'User:{user}, Curr_comb:{curr_comb}, Sensor:{sensor}, FDomain:{fdomain}')
                    gen_tr, gen_ts, imp_tr, imp_ts = Prep.get_data(DefaultParam.FEATUREFILE_PATH, fdomain, sensor, imposter_user_list, user, DefaultParam.NUM_SAMPLE_IMP)
                    comb_fdomain_gtr = pd.concat([comb_fdomain_gtr, gen_tr], axis=1)
                    comb_fdomain_gts = pd.concat([comb_fdomain_gts, gen_ts], axis=1)
                    comb_fdomain_itr = pd.concat([comb_fdomain_itr, imp_tr], axis=1)
                    comb_fdomain_its = pd.concat([comb_fdomain_its, imp_ts], axis=1)

            if 'gtr_fm' not in locals():  # If one exists all would do
                comb_sens_gtr = comb_fdomain_gtr
                comb_sens_gts = comb_fdomain_gts
                comb_sens_itr = comb_fdomain_itr
                comb_sens_its = comb_fdomain_its
            else:
                # damage control, just in case due to the error in sampling rate calculation the num of
                # rows varies across sensors
                min_sense_gtr = min(comb_sens_gtr.shape[0], comb_fdomain_gtr.shape[0])
                min_sense_gts = min(comb_sens_gts.shape[0], comb_fdomain_gts.shape[0])
                min_sense_itr = min(comb_sens_itr.shape[0], comb_fdomain_itr.shape[0])
                min_sense_its = min(comb_sens_its.shape[0], comb_fdomain_its.shape[0])

                comb_sens_gtr = pd.concat([comb_sens_gtr.iloc[0:min_sense_gtr, :], comb_fdomain_gtr.iloc[0:min_sense_gtr, :]], axis=1)
                comb_sens_gts = pd.concat([comb_sens_gts.iloc[0:min_sense_gts, :], comb_fdomain_gts.iloc[0:min_sense_gts, :]], axis=1)
                comb_sens_itr = pd.concat([comb_sens_itr.iloc[0:min_sense_itr, :], comb_fdomain_itr.iloc[0:min_sense_itr, :]], axis=1)
                comb_sens_its = pd.concat([comb_sens_its.iloc[0:min_sense_its, :], comb_fdomain_its.iloc[0:min_sense_its, :]], axis=1)

        # Saving the indices where sensors separate in feature matrix
            separating_indices.append(comb_sens_gtr.shape[1]) # saving the sensor separator indices
        all_feature_names  = comb_sens_gtr.columns

        # Remove outliers i.e. NaN or inf just in case
        comb_sens_gtr.fillna(comb_sens_gtr.median(), inplace=True)
        comb_sens_gts.fillna(comb_sens_gts.median(), inplace=True)
        comb_sens_itr.fillna(comb_sens_itr.median(), inplace=True)
        comb_sens_its.fillna(comb_sens_its.median(), inplace=True)

        # Perform feature normalization regardless what algorithms you would apply
        norm_comb_sens_gtr, norm_comb_sens_gts, norm_comb_sens_itr, norm_comb_sens_its = Prep.get_normalized_data_using_minmax(
                comb_sens_gtr, comb_sens_gts, comb_sens_itr, comb_sens_its)

        # Saving feature names before passing
        if DefaultParam.CLASS_BALANCING:
            bal_norm_comb_sens_gtr, bal_norm_comb_sens_gts, bal_norm_comb_sens_itr, bal_norm_comb_sens_its  = Prep.get_balanced_data(norm_comb_sens_gtr,
                                                                                                        norm_comb_sens_gts,
                                                                                                        norm_comb_sens_itr,
                                                                                                        norm_comb_sens_its)
        # Performing feature selection for each sensors individually and then combining them
        # This is for future if you would like to evaluate the effect of number of features
        # Currently, just use top 30/40 features from every sensor.
        for num_features in list_num_features:
            if 'comb_selected_sens_gtr' in locals():  # If one exists all would do
                del comb_selected_sens_gtr
                del comb_selected_sens_gts
                del comb_selected_sens_itr
                del comb_selected_sens_its

            for sens_itr in range(len(separating_indices)-1):
                # separating columns belonging to each sensors and selecting "num_features" features
                curr_sens_gtr = bal_norm_comb_sens_gtr[:,list(range(separating_indices[sens_itr], separating_indices[sens_itr+1],1))]
                curr_sens_gts = bal_norm_comb_sens_gts[:,list(range(separating_indices[sens_itr], separating_indices[sens_itr+1],1))]
                curr_sens_itr = bal_norm_comb_sens_itr[:,list(range(separating_indices[sens_itr], separating_indices[sens_itr+1],1))]
                curr_sens_its = bal_norm_comb_sens_its[:,list(range(separating_indices[sens_itr], separating_indices[sens_itr+1],1))]
                if 'comb_selected_sens_gtr' not in locals():  # If one exists all would do
                    comb_selected_sens_gtr, comb_selected_sens_gts, comb_selected_sens_itr, comb_selected_sens_its = fsselect(curr_sens_gtr, curr_sens_gts, curr_sens_itr,curr_sens_its, num_features)
                else:
                    temp_selected_sens_gtr, temp_selected_sens_gts, temp_selected_sens_itr, temp_selected_sens_its = fsselect(curr_sens_gtr, curr_sens_gts, curr_sens_itr,curr_sens_its, num_features)

                    comb_selected_sens_gtr = np.hstack((comb_selected_sens_gtr, temp_selected_sens_gtr))
                    comb_selected_sens_gts = np.hstack((comb_selected_sens_gts, temp_selected_sens_gts))
                    comb_selected_sens_itr = np.hstack((comb_selected_sens_itr, temp_selected_sens_itr))
                    comb_selected_sens_its = np.hstack((comb_selected_sens_its, temp_selected_sens_its))
            # print('comb_selected_sens_gtr',comb_selected_sens_gtr.shape)

        ## Training the model and finding the best parameter using 10 fold cross validation
            training_data = np.vstack((comb_selected_sens_gtr, comb_selected_sens_itr))
            training_labels = np.concatenate((Prep.get_labels(comb_selected_sens_gtr, 1), Prep.get_labels(comb_selected_sens_itr, -1)))
        # parameter preparation
            # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
            # use the random grid to search for best hyperparameters
            # first create the base model to tune
            scorerHTER = make_scorer(Utilities.scoringHTER, greater_is_better=False)
            # scorerFAR = make_scorer(Utilities.scoringFAR, greater_is_better=False)
            # scorerFRR = make_scorer(Utilities.scoringFRR, greater_is_better=False)
            # the distance metric of features to consider at every split
            n_neighbors = [int(x) for x in range(5, 9,1)]
            # print('n_neighbors',n_neighbors)
            dist_met = ['manhattan', 'euclidean']
            # create the random grid
            param_grid = {'n_neighbors': n_neighbors,
                          'metric': dist_met}
            knn_rand_model = KNeighborsClassifier()
            scoring_function = 'f1'
            knn_grid_search = GridSearchCV(estimator=knn_rand_model, param_grid=param_grid, cv=10, scoring=scorerHTER)
            knn_grid_search.fit(training_data, training_labels)
            best_nn = knn_grid_search.best_params_['n_neighbors']
            best_dist = knn_grid_search.best_params_['metric']

            # Retraining the model again and testing under different attack scenarios
            FinalModel = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist)
            FinalModel.fit(training_data, training_labels)

            # Training HTER:
            pred_gen_lables_tr = FinalModel.predict(comb_selected_sens_gtr)
            pred_imp_lables_tr = FinalModel.predict(comb_selected_sens_itr)
            pred_labels_tr = np.concatenate((pred_gen_lables_tr, pred_imp_lables_tr))
            actual_labels_tr = np.concatenate((Prep.get_labels(comb_selected_sens_gtr, 1), Prep.get_labels(comb_selected_sens_itr, -1)))
            tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(actual_labels_tr, pred_labels_tr).ravel()
            far_tr = fp_tr / (fp_tr + tn_tr)
            frr_tr = fn_tr / (fn_tr + tp_tr)
            hter_tr = (far_tr + frr_tr) / 2


            # Testing under zero-effort
            pred_gen_lables_zts = FinalModel.predict(comb_selected_sens_gts)
            pred_imp_lables_zts = FinalModel.predict(comb_selected_sens_its)
            pred_labels_zts = np.concatenate((pred_gen_lables_zts, pred_imp_lables_zts))
            actual_labels_zts = np.concatenate((Prep.get_labels(comb_selected_sens_gts, 1), Prep.get_labels(comb_selected_sens_its, -1)))

            # computing the error rates for the current predictions
            tn_zts, fp_zts, fn_zts, tp_zts = confusion_matrix(actual_labels_zts, pred_labels_zts).ravel()
            far_zts = fp_zts / (fp_zts + tn_zts)
            frr_zts = fn_zts / (fn_zts + tp_zts)
            hter_zts = (far_zts + frr_zts) / 2
            row_counter = row_counter + 1
        # https://pyformat.info/
        print(f'{user}: {curr_comb}, nn:{best_nn}, dist:{best_dist} | TRAIN_HTER({hter_tr:0.3f}) | FAR({far_zts:.3f}), FRR({frr_zts:0.3f}), TEST_HTER({hter_zts:.3f})')
        ulptable_zts.loc[row_counter] = [user, curr_comb, best_nn, best_dist, far_zts, frr_zts, hter_zts, hter_tr]
    cuser_df = ulptable_zts[ulptable_zts.User == user]
    cuser_df.to_csv(os.path.join(result_location, user + '.csv'))

# Writing the full results
ulptable_zts.to_csv(os.path.join(result_location, os.path.basename(__file__)[:-3] + 'full_results.csv'))
print(ulptable_zts.to_string())
