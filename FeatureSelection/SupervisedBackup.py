import os
import sys
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif
# Adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from Preprocess import Prep
from GlobalParameters import DefaultParam
from sklearn.feature_selection import SelectFromModel

#___________________________________________________________________________________________________________________#
# Works on ndarray
def fs_using_randfor(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data,fsthreshold): # fsthreshold is [0, 3]
    clf = ExtraTreesClassifier(n_estimators=300, max_features='log2', max_depth=10)
    # Training data does not contain labels explicitly as we can get label anytime
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    train_labels = np.concatenate((Prep.get_labels(gen_tr_data, 1), Prep.get_labels(imp_tr_data, -1)))
    #### Filtering 50% features using statistical tests before selecting them
    # print('Before chi square test.csv training_data.shape:', gen_tr_data.shape)
    ChiSTestModel = SelectPercentile(mutual_info_classif, percentile=70).fit(training_data, train_labels)
    gen_tr_data = ChiSTestModel.transform(gen_tr_data)
    gen_ts_data = ChiSTestModel.transform(gen_ts_data)
    imp_tr_data = ChiSTestModel.transform(imp_tr_data)
    imp_ts_data = ChiSTestModel.transform(imp_ts_data)
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    # print('After chi square test.csv training_data.shape:', gen_tr_data.shape)
    #Applying the random forest based feature ranking
    clf = clf.fit(training_data, train_labels)
    # print('clf.feature_importances_',clf.feature_importances_)
    RFBasedModel = SelectFromModel(clf, prefit=True, threshold=str(fsthreshold)+'*median')
    selected_feature_indices = RFBasedModel.get_support()
    gen_tr_data = RFBasedModel.transform(gen_tr_data)
    gen_ts_data = RFBasedModel.transform(gen_ts_data)
    imp_tr_data = RFBasedModel.transform(imp_tr_data)
    imp_ts_data = RFBasedModel.transform(imp_ts_data)
    # print('After applying the RFBased feature selection:', gen_tr_data.shape)
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, selected_feature_indices

#___________________________________________________________________________________________________________________#
# LDA Tranform returns k-1 components.. k is the number of classes
# So the number of components for LDA would be 1 out case as # there are two classes
# LDA is used WHEN feature is CONTINUOUS and CLass is Categorical ..
# But we have only two classes
# https://www.datacamp.com/community/tutorials/feature-selection-python
# Works on ndarray
def fs_using_lda(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, fsthreshold):
    num_component = int(DefaultParam.fsthreshold * gen_tr_data.shape[1])
    # print('num_component',num_component)
    lda = LinearDiscriminantAnalysis(n_components=num_component)
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    training_labels = np.concatenate((Prep.get_labels(gen_tr_data, 1), Prep.get_labels(imp_tr_data, -1)))
    lda = lda.fit(training_data, training_labels)
    gen_tr_data = lda.transform(gen_tr_data)
    gen_ts_data = lda.transform(gen_ts_data)
    imp_tr_data = lda.transform(imp_tr_data)
    imp_ts_data = lda.transform(imp_ts_data)
    # print('Printing the training and testing samples inside LDA')
    # print(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data)
    # Returning indices of featues does not apply here as LDA returns transformed component not the features or indices
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data

#___________________________________________________________________________________________________________________#
# Works on ndarray
def fs_using_FCBF(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, fsthreshold):
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    train_labels = np.concatenate((Prep.get_labels(gen_tr_data, 1), Prep.get_labels(imp_tr_data, -1)))
    # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    selected_feature_indices = fcbf(training_data, train_labels, fsthreshold)  # threshold is between 0 to 1 SU is between zero to one
    # print(f'selected features:{good_features}')
    gen_tr_data = gen_tr_data.loc[:, selected_feature_indices]
    gen_ts_data = gen_ts_data.loc[:, selected_feature_indices]
    imp_tr_data = imp_tr_data.loc[:, selected_feature_indices]
    imp_ts_data = imp_ts_data.loc[:, selected_feature_indices]
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, selected_feature_indices

# !/usr/bin/env python
# https://github.com/shiralkarprashant/FCBF/blob/master/src/fcbf.py"""
# fcbf.py
# Created by Prashant Shiralkar on 2015-02-06.
# Fast Correlation-Based Filter (FCBF) algorithm as described in
# Feature Selection for High-Dimensional Data: A Fast Correlation-Based
# Filter Solution. Yu & Liu (ICML 2003)
def entropy(vec, base=2):
    " Returns the empirical entropy H(X) in the input vector."
    _, vec = np.unique(vec, return_counts=True)
    prob_vec = np.array(vec / float(sum(vec)))
    if base == 2:
        logfn = np.log2
    elif base == 10:
        logfn = np.log10
    else:
        logfn = np.log
    return prob_vec.dot(-logfn(prob_vec))

def conditional_entropy(x, y):
    "Returns H(X|Y)."
    uy, uyc = np.unique(y, return_counts=True)
    prob_uyc = uyc / float(sum(uyc))
    cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
    return prob_uyc.dot(cond_entropy_x)

def mutual_information(x, y):
    " Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
    return entropy(x) - conditional_entropy(x, y)

def symmetrical_uncertainty(x, y):
    " Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
    return 2.0 * mutual_information(x, y) / (entropy(x) + entropy(y))

def getFirstElement(d):
    """
    Returns tuple corresponding to first 'unconsidered' feature

    Parameters:
    __________--
    d : ndarray
        A 2-d array with SU, original feature index and flag as columns.

    Returns:
    _____---
    a, b, c : tuple
        a - SU value, b - original feature index, c - index of next 'unconsidered' feature
    """

    t = np.where(d[:, 2] > 0)[0]
    if len(t):
        return d[t[0], 0], d[t[0], 1], t[0]
    return None, None, None


def getNextElement(d, idx):
    """
    Returns tuple corresponding to the next 'unconsidered' feature.

    Parameters:
    __________---
    d : ndarray
        A 2-d array with SU, original feature index and flag as columns.
    idx : int
        Represents original index of a feature whose next element is required.

    Returns:
    __________
    a, b, c : tuple
        a - SU value, b - original feature index, c - index of next 'unconsidered' feature
    """
    t = np.where(d[:, 2] > 0)[0]
    t = t[t > idx]
    if len(t):
        return d[t[0], 0], d[t[0], 1], t[0]
    return None, None, None

def removeElement(d, idx):
    """
    Returns data with requested feature removed.

    Parameters:
    __________---
    d : ndarray
        A 2-d array with SU, original feature index and flag as columns.
    idx : int
        Represents original index of a feature which needs to be removed.

    Returns:
    __________
    d : ndarray
        Same as input, except with specific feature removed.
    """
    d[idx, 2] = 0
    return d


def c_correlation(X, y):
    """
    Returns SU values between each feature and class.

    Parameters:
    __________---
    X : 2-D ndarray
        Feature matrix.
    y : ndarray
        Class label vector

    Returns:
    __________
    su : ndarray
        Symmetric Uncertainty (SU) values for each feature.
    """
    su = np.zeros(X.shape[1])
    for i in np.arange(X.shape[1]):
        su[i] = symmetrical_uncertainty(X[:, i], y)
    return su


def fcbf(X, y, thresh):
    """
    Perform Fast Correlation-Based Filter solution (FCBF).

    Parameters:
    __________---
    X : 2-D ndarray
        Feature matrix
    y : ndarray
        Class label vector
    thresh : float
        A value in [0,1) used as threshold for selecting 'relevant' features.
        A negative value suggest the use of minimum SU[i,c] value as threshold.

    Returns:
    __________
    sbest : 2-D ndarray
        An array containing SU[i,c] values and feature index i.
    """
    n = X.shape[1]
    slist = np.zeros((n, 3))
    slist[:, -1] = 1

    # identify relevant features
    slist[:, 0] = c_correlation(X, y)  # compute 'C-correlation'
    idx = slist[:, 0].argsort()[::-1]
    slist = slist[idx,]
    slist[:, 1] = idx
    if thresh < 0:
        thresh = np.median(slist[-1, 0])
        print("Using minimum SU value as default threshold: {0}".format(thresh))
    elif thresh >= 1 or thresh > max(slist[:, 0]):
        raise ValueError("No relevant features selected for given threshold.")
    slist = slist[slist[:, 0] > thresh, :]  # desc. ordered per SU[i,c]

    # identify redundant features among the relevant ones
    cache = {}
    m = len(slist)
    p_su, p, p_idx = getFirstElement(slist)
    for i in np.xrange(m):
        p = int(p)
        q_su, q, q_idx = getNextElement(slist, p_idx)
        if q:
            while q:
                q = int(q)
                if (p, q) in cache:
                    pq_su = cache[(p, q)]
                else:
                    pq_su = symmetrical_uncertainty(X[:, p], X[:, q])
                    cache[(p, q)] = pq_su

                if pq_su >= q_su:
                    slist = removeElement(slist, q_idx)
                q_su, q, q_idx = getNextElement(slist, q_idx)

        p_su, p, p_idx = getNextElement(slist, p_idx)
        if not p_idx:
            break

    sbest = slist[slist[:, 2] > 0, :2]
    return sbest

# Another source of implementation is https://github.com/SantiagoEG/FCBF_module
    