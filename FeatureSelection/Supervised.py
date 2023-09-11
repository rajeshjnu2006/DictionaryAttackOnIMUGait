import os
import sys
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
# Adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from Preprocess import Prep
from GlobalParameters import DefaultParam
from sklearn.feature_selection import SelectPercentile, chi2,SelectKBest

#___________________________________________________________________________________________________________________#
# Works on ndarray
def fs_using_MI(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data,fsthreshold):
    # fsthreshold is [0, 3],...., column_names may be used for deeper analysis
    fselector = SelectKBest(mutual_info_classif, k=int(fsthreshold))
    # Training data does not contain labels explicitly as we can get label anytime
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    train_labels = np.concatenate((Prep.get_labels(gen_tr_data, 1), Prep.get_labels(imp_tr_data, -1)))
    fselector.fit(training_data,train_labels)
    gen_tr_data = fselector.transform(gen_tr_data)
    gen_ts_data = fselector.transform(gen_ts_data)
    imp_tr_data = fselector.transform(imp_tr_data)
    imp_ts_data = fselector.transform(imp_ts_data)
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, fselector

#___________________________________________________________________________________________________________________#

# Works on ndarray
def fs_using_ChiMI(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data,fsthreshold):
    # Training data does not contain labels explicitly as we can get label anytime
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    train_labels = np.concatenate((Prep.get_labels(gen_tr_data, 1), Prep.get_labels(imp_tr_data, -1)))
    fselector_chi2 = SelectPercentile(chi2, percentile=60).fit(training_data, train_labels)
    # print('Before chi square test.csv training_data.shape:', gen_tr_data.shape)
    gen_tr_data = fselector_chi2.transform(gen_tr_data)
    gen_ts_data = fselector_chi2.transform(gen_ts_data)
    imp_tr_data = fselector_chi2.transform(imp_tr_data)
    imp_ts_data = fselector_chi2.transform(imp_ts_data)

    training_data = np.vstack((gen_tr_data, imp_tr_data))
    # print('After chi square test.csv training_data.shape:', gen_tr_data.shape)
    # Applying the random forest based feature ranking
    fselector = SelectKBest(mutual_info_classif, k=int(fsthreshold))
    fselector.fit(training_data,train_labels)
    # print('Before chi square test.csv training_data.shape:', gen_tr_data.shape)
    gen_tr_data = fselector.transform(gen_tr_data)
    gen_ts_data = fselector.transform(gen_ts_data)
    imp_tr_data = fselector.transform(imp_tr_data)
    imp_ts_data = fselector.transform(imp_ts_data)
    # print('After applying the RFBased feature selection:', gen_tr_data.shape)
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data
#___________________________________________________________________________________________________________________#

# LDA Tranform returns k-1 components.. k is the number of classes
# So the number of components for LDA would be 1 out case as # there are two classes
# LDA is used WHEN feature is CONTINUOUS and CLass is Categorical ..
# But we have only two classes
# https://www.datacamp.com/community/tutorials/feature-selection-python
# Works on ndarray
def ft_using_lda(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, fsthreshold):
    num_component = int(fsthreshold * gen_tr_data.shape[1])
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
def fs_using_FCBF(gen_tr_data, imp_tr_data, fsthreshold, column_names): # column_names maybe used for deeper analysis
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    train_labels = np.concatenate((Prep.get_labels(gen_tr_data, 1), Prep.get_labels(imp_tr_data, -1)))
    # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    # creating an object
    FCBFObj = FCBF(fsthreshold)
    FCBFObj.fit(training_data, train_labels)
    selected_feature_indices = FCBFObj.idx_sel  # threshold is between 0 to 1 SU is between zero to one
    return selected_feature_indices


#---- Helper functions
def count_vals(x):
    vals = np.unique(x)
    occ = np.zeros(shape=vals.shape)
    for i in range(vals.size):
        occ[i] = np.sum(x == vals[i])
    return occ


def entropy(x):
    n = float(x.shape[0])
    ocurrence = count_vals(x)
    px = ocurrence / n
    return -1 * np.sum(px * np.log2(px))


def symmetricalUncertain(x, y):
    n = float(y.shape[0])
    vals = np.unique(y)
    # Computing Entropy for the feature x.
    Hx = entropy(x)
    # Computing Entropy for the feature y.
    Hy = entropy(y)
    # Computing Joint entropy between x and y.
    partial = np.zeros(shape=(vals.shape[0]))
    for i in range(vals.shape[0]):
        partial[i] = entropy(x[y == vals[i]])

    partial[np.isnan(partial) == 1] = 0
    py = count_vals(y).astype(dtype='float64') / n
    Hxy = np.sum(py[py > 0] * partial)
    IG = Hx - Hxy
    return 2 * IG / (Hx + Hy)


def suGroup(x, n):
    m = x.shape[0]
    x = np.reshape(x, (n, m / n)).T
    m = x.shape[1]
    SU_matrix = np.zeros(shape=(m, m))
    for j in range(m - 1):
        x2 = x[:, j + 1::]
        y = x[:, j]
        temp = np.apply_along_axis(symmetricalUncertain, 0, x2, y)
        for k in range(temp.shape[0]):
            SU_matrix[j, j + 1::] = temp
            SU_matrix[j + 1::, j] = temp

    return 1 / float(m - 1) * np.sum(SU_matrix, axis=1)


def isprime(a):
    return all(a % i for i in range(2, a))


"""
get
"""


def get_i(a):
    if isprime(a):
        a -= 1
    return filter(lambda x: a % x == 0, range(2, a))


"""
FCBF - Fast Correlation Based Filter
L. Yu and H. Liu. Feature Selection for High‐Dimensional Data: A Fast Correlation‐Based Filter Solution. 
In Proceedings of The Twentieth International Conference on Machine Leaning (ICML‐03), 856‐863.
Washington, D.C., August 21‐24, 2003.
"""


class FCBF:
    idx_sel = []

    def __init__(self, th=0.01):
        '''
        Parameters
        ---------------
            th = The initial threshold
        '''
        self.th = th

    def fit(self, x, y):
        '''
        This function executes FCBF algorithm and saves indexes
        of selected features in self.idx_sel

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.idx_sel = []
        """
        First Stage: Computing the SU for each feature with the response.
        """
        SU_vec = np.apply_along_axis(symmetricalUncertain, 0, x, y)
        SU_list = SU_vec[SU_vec > self.th]
        SU_list[::-1].sort()

        m = x[:, SU_vec > self.th].shape
        x_sorted = np.zeros(shape=m)

        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = 0
            x_sorted[:, i] = x[:, ind].copy()
            self.idx_sel.append(ind)

        """
        Second Stage: Identify relationships between feature to remove redundancy.
        """
        j = 0
        while True:
            """
            Stopping Criteria:The search finishes
            """
            if j >= x_sorted.shape[1]: break
            y = x_sorted[:, j].copy()
            x_list = x_sorted[:, j + 1:].copy()
            if x_list.shape[1] == 0: break

            SU_list_2 = SU_list[j + 1:]
            SU_x = np.apply_along_axis(symmetricalUncertain, 0,
                                       x_list, y)

            comp_SU = SU_x >= SU_list_2
            to_remove = np.where(comp_SU)[0] + j + 1
            if to_remove.size > 0:
                x_sorted = np.delete(x_sorted, to_remove, axis=1)
                SU_list = np.delete(SU_list, to_remove, axis=0)
                to_remove.sort()
                for r in reversed(to_remove):
                    self.idx_sel.remove(self.idx_sel[r])
            j = j + 1

    def fit_transform(self, x, y):
        '''
        This function fits the feature selection
        algorithm and returns the resulting subset.

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.fit(x, y)
        return x[:, self.idx_sel]

    def transform(self, x):
        '''
        This function applies the selection
        to the vector x.

        Parameters
        ---------------
            x = dataset  [NxM]
        '''
        return x[:, self.idx_sel]


"""
FCBF# - Fast Correlation Based Filter 
B. Senliol, G. Gulgezen, et al. Fast Correlation Based Filter (FCBF) with a Different Search Strategy. 
In Computer and Information Sciences (ISCIS ‘08) 23rd International Symposium on, pages 1‐4. 
Istanbul, October 27‐29, 2008.
"""


class FCBFK(FCBF):
    idx_sel = []

    def __init__(self, k=10):
        '''
        Parameters
        ---------------
            k = Number of features to include in the
            subset.
        '''
        self.k = k

    def fit(self, x, y):
        '''
        This function executes FCBFK algorithm and saves indexes
        of selected features in self.idx_sel

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.idx_sel = []
        """
        First Stage: Computing the SU for each feature with the response.
        """
        SU_vec = np.apply_along_axis(symmetricalUncertain, 0, x, y)

        SU_list = SU_vec[SU_vec > 0]
        SU_list[::-1].sort()

        m = x[:, SU_vec > 0].shape
        x_sorted = np.zeros(shape=m)

        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = 0
            x_sorted[:, i] = x[:, ind].copy()
            self.idx_sel.append(ind)

        """
        Second Stage: Identify relationships between features to remove redundancy with stopping 
        criteria (features in x_best == k).
        """
        j = 0
        while True:
            y = x_sorted[:, j].copy()
            SU_list_2 = SU_list[j + 1:]
            x_list = x_sorted[:, j + 1:].copy()

            """
            Stopping Criteria:The search finishes
            """
            if x_list.shape[1] == 0: break

            SU_x = np.apply_along_axis(symmetricalUncertain, 0,
                                       x_list, y)

            comp_SU = SU_x >= SU_list_2
            to_remove = np.where(comp_SU)[0] + j + 1
            if to_remove.size > 0 and x.shape[1] > self.k:

                for i in reversed(to_remove):

                    x_sorted = np.delete(x_sorted, i, axis=1)
                    SU_list = np.delete(SU_list, i, axis=0)
                    self.idx_sel.remove(self.idx_sel[i])
                    if x_sorted.shape[1] == self.k: break

            if x_list.shape[1] == 1 or x_sorted.shape[1] == self.k:
                break
            j = j + 1

        if len(self.idx_sel) > self.k:
            self.idx_sel = self.idx_sel[:self.k]


"""
FCBFiP - Fast Correlation Based Filter in Pieces
"""


class FCBFiP(FCBF):
    idx_sel = []

    def __init__(self, k=10, npieces=2):
        '''
        Parameters
        ---------------
            k = Number of features to include in the
            subset.
            npieces = Number of pieces to divide the
            feature space.
        '''
        self.k = k
        self.npieces = npieces

    def fit(self, x, y):
        '''
        This function executes FCBF algorithm and saves indexes
        of selected features in self.idx_sel

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''

        """
        First Stage: Computing the SU for each feature with the response. We sort the 
        features. When we have a prime number of features we remove the last one from the
        sorted features list.
        """
        m = x.shape
        nfeaturesPieces = int(m[1] / float(self.npieces))
        SU_vec = np.apply_along_axis(symmetricalUncertain, 0, x, y)

        x_sorted = np.zeros(shape=m, dtype='float64')
        idx_sorted = np.zeros(shape=m[1], dtype='int64')
        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = -1
            idx_sorted[i] = ind
            x_sorted[:, i] = x[:, ind].copy()

        if isprime(m[1]):
            x_sorted = np.delete(x_sorted, m[1] - 1, axis=1)
            ind_prime = idx_sorted[m[1] - 1]
            idx_sorted = np.delete(idx_sorted, m[1] - 1)
            # m = x_sorted.shape
        """
        Second Stage: Identify relationships between features into its vecinity
        to remove redundancy with stopping criteria (features in x_best == k).
        """

        x_2d = np.reshape(x_sorted.T, (self.npieces, nfeaturesPieces * m[0])).T

        SU_x = np.apply_along_axis(suGroup, 0, x_2d, nfeaturesPieces)
        SU_x = np.reshape(SU_x.T, (self.npieces * nfeaturesPieces,))
        idx_sorted2 = np.zeros(shape=idx_sorted.shape, dtype='int64')
        SU_x[np.isnan(SU_x)] = 1

        for i in range(idx_sorted.shape[0]):
            ind = np.argmin(SU_x)
            idx_sorted2[i] = idx_sorted[ind]
            SU_x[ind] = 10

        """
        Scoring step
        """
        self.scores = np.zeros(shape=m[1], dtype='int64')

        for i in range(m[1]):
            if i in idx_sorted:
                self.scores[i] = np.argwhere(i == idx_sorted) + np.argwhere(i == idx_sorted2)
        if isprime(m[1]):
            self.scores[ind_prime] = 2 * m[1]
        self.set_k(self.k)

    def set_k(self, k):
        self.k = k
        scores_temp = -1 * self.scores

        self.idx_sel = np.zeros(shape=self.k, dtype='int64')
        for i in range(self.k):
            ind = np.argmax(scores_temp)
            scores_temp[ind] = -100000000
            self.idx_sel[i] = ind
# Another source of implementation is https://github.com/SantiagoEG/FCBF_module
    