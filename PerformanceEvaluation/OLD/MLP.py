#!/usr/bin/env python -W ignore::DeprecationWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import warnings
import numpy as np
import pandas as pd
import os
import sys
# adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
from sklearn.neural_network import MLPClassifier

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
ulptable = pd.DataFrame(columns=['User', 'Sensors', 'Num_features', 'hlayer_size', 'activation', 'solver', 'alpha', 'learning_rate','Training_score','FAR', 'FRR', 'HTER'])
row_counter = 0
# create a folder to store the results
result_location = os.path.join(os.getcwd(),os.path.basename(__file__)[:-3]+'Results')
if os.path.exists(result_location):
    # raise FileExistsError("#_________________The result directory already exists.. delete manually")
    print('removing and creating a new folder')
else:
    os.mkdir(result_location)
    print(f'#_________________Created a new directory to save results at {result_location}')
#___________________________________________________________________________________________________________________#
if DefaultParam.CLASS_BALANCING:
    print('#_________________Applying SMOTE for CLASS BALANCING')

#___________________________________________________________________________________________________________________#
# DefaultParam.FEATURE_DOMAIN_LIST = ['FrequencyFeatures', 'ITheoryFeatures', 'TimeFeatures']
if DefaultParam.FEATURE_DOMAIN_LIST:
    print('#_________________Using features from',DefaultParam.FEATURE_DOMAIN_LIST)
# using this first time and feeling very good about it .. function as first class object
# fsselect = Unsupervised.ft_using_pca
fsselect = Supervised.fs_using_MI
fsthresholds = np.linspace(start=30, stop=60, num=7)
fsthresholds = [int(item) for item in fsthresholds]


# gen_user_list = ['User9', 'User11']
#___________________________________________________________#
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
        # with open('listfile.txt', 'w') as filehandle:
        #     filehandle.writelines("%s\n" % place for place in feature_names)
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
            training_data = np.vstack((com_gen_tr_data_selected, com_imp_tr_data_selected))
            training_labels = np.concatenate(
                (Prep.get_labels(com_gen_tr_data_selected, 1), Prep.get_labels(com_imp_tr_data_selected, -1)))
            hlsize = []
            for flayer in range(150, 251, 20):
                # hlsize.append((flayer, ))
                for slayer in range(100, 101, 50):
                    hlsize.append((flayer, slayer))
            # solver = ['sgd', 'adam', 'lbfgs']
            solver = ['adam']  # getting a feel
            # activation = ['identity', 'tanh', 'relu', 'logistic']
            activation = ['relu']  # getting a feel
            # Read: "https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html"
            # Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by
            # constraining the size of the weights.Increasing alpha may fix high variance (a sign of overfitting) by
            # encouraging smaller weights, resulting in a decision boundary plot that appears with lesser
            # curvatures.Similarly, decreasing alpha may fix high bias (a sign of underfitting) by encouraging larger
            # weights, potentially resulting in a more complicated decision boundary.
            alpha = [0.01, 0.1,
                     1.0]  # default is 0.0001, # Experiment with this three values.. higher the alpha, simpler the boundary
            alpha = [1.0]
            learning_rate = ['adaptive']
            # Create the random grid
            param_grid = {'hidden_layer_sizes': hlsize,
                          'activation': activation,
                          'solver': solver,
                          'alpha': alpha,
                          'learning_rate': learning_rate}
            # Create a based model
            MLPGridModel = MLPClassifier(max_iter=1000, random_state=DefaultParam.RANDOM_SEED, early_stopping=True)
            # Instantiate the grid search model
            scorerHTER = make_scorer(Utilities.scoringHTER, greater_is_better=False)
            # scorerFAR = make_scorer(Utilities.scoringFAR, greater_is_better=False)
            # scorerFRR = make_scorer(Utilities.scoringFRR, greater_is_better=False)
            scoring_function = 'f1'
            reverse_order = False # Make it False if using FAR, FRR, HETER as scoring functions
            MLPGSearch = GridSearchCV(estimator=MLPGridModel, param_grid=param_grid, cv=10, n_jobs=-1,
                                       verbose=0, scoring='f1')
            MLPGSearch.fit(training_data, training_labels)
            hlayers_gp = MLPGSearch.best_params_['hidden_layer_sizes']
            activation_gp = MLPGSearch.best_params_['activation']
            solver_gp = MLPGSearch.best_params_['solver']
            alpha_gp = MLPGSearch.best_params_['alpha']
            learning_rate_gp = MLPGSearch.best_params_['learning_rate']
            Training_stats.append([fs_threshold, hlayers_gp, activation_gp, solver_gp, alpha_gp, learning_rate_gp, MLPGSearch.best_score_])

        best_params = Utilities.get_best_params_median_features(Training_stats, reverse_order)
        # To consider in future: there might be many params setting with the best
        best_fstheshold = best_params[0]
        best_hlayers_gp = best_params[1]
        best_activation_gp = best_params[2]
        best_solver_gp = best_params[3]
        best_alpha_gp = best_params[4]
        best_learning_rate_gp = best_params[5]
        best_training_score = best_params[6]

        # retraining the model using the best parametersknn found from the training dataset
        # To retrain, selecting the features again using the best threshold based on the training score
        com_gen_tr_data_selected, com_gen_ts_data_selected, com_imp_tr_data_selected, com_imp_ts_data_selected = fsselect(
            com_gen_tr_data, com_gen_ts_data, com_imp_tr_data,
            com_imp_ts_data, best_fstheshold)

        training_data = np.vstack((com_gen_tr_data_selected, com_imp_tr_data_selected))
        training_labels = np.concatenate(
            (Prep.get_labels(com_gen_tr_data_selected, 1), Prep.get_labels(com_imp_tr_data_selected, -1)))
        finalmodel = MLPClassifier(max_iter=1000, hidden_layer_sizes=best_hlayers_gp, activation=best_activation_gp,
                                   random_state=DefaultParam.RANDOM_SEED,
                                   solver=best_solver_gp, alpha=best_alpha_gp, learning_rate=best_learning_rate_gp)

        finalmodel.fit(training_data, training_labels)
        # testing for the Genuine
        pred_genuine_lables = finalmodel.predict(com_gen_ts_data_selected)
        # testing for the impostors
        pred_impostor_lables = finalmodel.predict(com_imp_ts_data_selected)
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
        # print('the best parameters:', grid.best_params_)
        print(
            f'{user} with sensors:{curr_comb}, features: {best_fstheshold}, hlayer:{best_hlayers_gp}, act:{best_activation_gp}, solver:{best_solver_gp}, alpha:{best_alpha_gp}, learning rate:{best_learning_rate_gp}, training score:{best_training_score},far:{far:.3f}, frr:{frr:0.3f} hter:{hter:0.3f}')
        ulptable.loc[row_counter] = [user, curr_comb, best_fstheshold, best_hlayers_gp, best_activation_gp, best_solver_gp, best_alpha_gp, best_learning_rate_gp, best_training_score, far, frr, hter]
    cuser_df = ulptable[ulptable.User == user]
    cuser_df.to_csv(os.path.join(result_location, user + '.csv'))

# Writing the full results
ulptable.to_csv(os.path.join(result_location,os.path.basename(__file__)[:-3]+'full_results.csv'))
