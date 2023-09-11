import numpy as np
import pandas as pd
from GlobalParameters import DefaultParam
from pyentrp import entropy as ent

# https://github.com/nikdon/pyEntropy
ITDFeatureDictionary = {}


def getShannonEntropy(x):
    max_bins = 10
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    return - np.sum(p * np.math.log(p) for p in probs if p != 0)


def getSampleEntropy(x, sample_length, tolerance):
    """
    Calculate and return sample entropy of x.

    .. rubric:: References

    |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
    |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract

    :param x: the time series to calculate the feature of
    :type x: pandas.Series

    :return: the value of this feature
    :return type: float
    """
    x = np.array(x)

    # sample_length = 1 # number of sequential points of the time series
    # tolerance = 0.2 * np.std(x) # 0.2 is a common value for r - why?

    n = len(x)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((1, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((1, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = x[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[0]))

    # sample entropy = -1 * (log (A/B))
    similarity_ratio = A / B
    se = -1 * np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se[0]


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series

    Args:
        time_series: Time series
        scale: Scale factor

    Returns:
        Vector of coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    temp = np.reshape(time_series[0:b * scale], (b, scale))
    cts = np.mean(temp, axis=1)
    return cts


def getMultiscaleEntropy(time_series, sample_length, tolerance=None, maxscale=None):
    """Calculate the Multiscale Entropy of the given time series considering
    different time-scales of the time series.

    Args:
        time_series: Time series for analysis
        sample_length: Bandwidth or group of points
        tolerance: Tolerance (default = 0.1*std(time_series))

    Returns:
        Vector containing Multiscale Entropy

    Reference:
        [1] http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html
    """

    if tolerance is None:
        # we need to fix the tolerance at this level. If it remains 'None' it will be changed in call to sample_entropy()
        tolerance = 0.1 * np.std(time_series)
    if maxscale is None:
        maxscale = len(time_series)

    mse = []

    for i in range(maxscale):
        temp = util_granulate_time_series(time_series, i + 1)
        mse.append(getSampleEntropy(temp, sample_length, tolerance))

    return mse


def getSampleEntropyForMSE(time_series, sample_length, tolerance=None):
    """Calculates the sample entropy of degree m of a time_series.

    This method uses chebychev norm.
    It is quite fast for random data, but can be slower is there is
    structure in the input time series.

    Args:
        time_series: numpy array of time series
        sample_length: length of longest template vector
        tolerance: tolerance (defaults to 0.1 * std(time_series)))
    Returns:
        Array of sample entropies:
            SE[k] is ratio "#templates of length k+1" / "#templates of length k"
            where #templates of length 0" = n*(n - 1) / 2, by definition
    Note:
        The parameter 'sample_length' is equal to m + 1 in Ref[1].


    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
            """
    # The code below follows the sample length convention of Ref [1] so:
    M = sample_length - 1

    time_series = np.array(time_series)
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)

    # Ntemp is a vector that holds the number of matches. N[k] holds matches templates of length k
    Ntemp = np.zeros(M + 2)
    # Templates of length 0 matches by definition:
    Ntemp[0] = n * (n - 1) / 2

    for i in range(n - M - 1):
        template = time_series[i:(i + M + 1)]  # We have 'M+1' elements in the template
        rem_time_series = time_series[i + 1:]

        searchlist = np.nonzero(np.abs(rem_time_series - template[0]) < tolerance)[0]

        go = len(searchlist) > 0

        length = 1

        Ntemp[length] += len(searchlist)

        while go:
            length += 1
            nextindxlist = searchlist + 1
            nextindxlist = nextindxlist[nextindxlist < n - 1 - i]  # Remove candidates too close to the end
            nextcandidates = rem_time_series[nextindxlist]
            hitlist = np.abs(nextcandidates - template[length - 1]) < tolerance
            searchlist = nextindxlist[hitlist]

            Ntemp[length] += np.sum(hitlist)

            go = any(hitlist) and length < M + 1

    sampen = - np.log(Ntemp[1:] / Ntemp[:-1])

    return sampen


def getCompositeMultiscaleEntropy(time_series, sample_length, scale, tolerance=None):
    """Calculate the Composite Multiscale Entropy of the given time series.

    Args:
        time_series: Time series for analysis
        sample_length: Number of sequential points of the time series
        scale: Scale factor
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))

    Returns:
        Vector containing Composite Multiscale Entropy

    Reference:
        [1] Wu, Shuen-De, et al. "Time series analysis using
            composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.
    """
    listmse = [0] * scale

    for i in range(scale):
        for j in range(i):
            tmp = util_granulate_time_series(time_series[j:], i + 1)
            listmse[i] += listmse[i]
            listmse[i] += getSampleEntropy(tmp, sample_length, tolerance) / (i + 1)
    listmse.pop()  # the first item was always zero so dropped it
    return listmse


# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/pyeeg/PyEEG_Ref_Guide.pdf
def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's difference function.

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if D is None:
        D = np.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))


# https://www.hindawi.com/journals/cin/2011/406391/
# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/pyeeg/PyEEG_Ref_Guide.pdf
def hfd(X):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = len(X)
    Kmax = int(N / 20)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=-1)
    return p[0]


# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/pyeeg/PyEEG_Ref_Guide.pdf
def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    ParameterskNN
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity


def embed_seq(X, Tau, D):
    """Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    ParameterskNN
    ----------

    X
        list

        a time series

    Tau
        integer

        the lag or delay when building embedding sequence

    D
        integer

        the embedding dimension

    Returns
    -------

    Y
        2-D list

        embedding matrix built

    Examples
    ---------------
    #>>> import pyeeg
   # >>> a=range(0,9)
   # >>> pyeeg.embed_seq(a,1,4)
    array([[ 0.,  1.,  2.,  3.],
           [ 1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.],
           [ 3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.],
           [ 5.,  6.,  7.,  8.]])
   # >>> pyeeg.embed_seq(a,2,3)
    array([[ 0.,  2.,  4.],
           [ 1.,  3.,  5.],
           [ 2.,  4.,  6.],
           [ 3.,  5.,  7.],
           [ 4.,  6.,  8.]])
   # >>> pyeeg.embed_seq(a,4,1)
    array([[ 0.],
           [ 1.],
           [ 2.],
           [ 3.],
           [ 4.],
           [ 5.],
           [ 6.],
           [ 7.],
           [ 8.]])

    """
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/pyeeg/PyEEG_Ref_Guide.pdf
def svd_entropy(X, W=None):
    """Compute SVD Entropy from either two cases below:
    1. a time series X, with lag tau and embedding dimension dE (default)
    2. a list, W, of normalized singular values of a matrix (if W is provided,
    recommend to speed up.)

    If W is None, the function will do as follows to prepare singular spectrum:

        First, computer an embedding matrix from X, Tau and DE using pyeeg
        function embed_seq():
                    M = embed_seq(X, Tau, DE)

        Second, use scipy.linalg function svd to decompose the embedding matrix
        M and obtain a list of singular values:
                    W = svd(M, compute_uv=0)

        At last, normalize W:
                    W /= sum(W)

    Notes
    -------------

    To speed up, it is recommended to compute W before calling this function
    because W may also be used by other functions whereas computing it here
    again will slow down.
    """
    Tau = 2
    DE = 10
    if W is None:
        Y = embed_seq(X, Tau, DE)
        W = np.linalg.svd(Y, compute_uv=0)
        W /= sum(W)  # normalize singular values

    return -1 * sum(W * np.log(W))


ITDFeatureDictionary['pfd'] = pfd
ITDFeatureDictionary['hfd'] = hfd
ITDFeatureDictionary['hjorth'] = hjorth
ITDFeatureDictionary['svd_entropy'] = svd_entropy

feature_values = []


def extract_itd_feature(series):
    feature_values = []
    feature_names = []
    for fname, function in ITDFeatureDictionary.items():
        # print(fname)
        if fname == 'hjorth':
            temp = ITDFeatureDictionary[fname](series)
            feature_values = feature_values + list(temp)
            for i, v in enumerate(temp):
                if i ==0:
                    feature_names.append('hjorth'+'_mobility')
                elif i == 1:
                    feature_names.append('hjorth'+'_complexity')
                else:
                    raise ValueError("hjorth has more than 2 features you need to handle it")
        else:
            feature_values.append(ITDFeatureDictionary[fname](series))
            feature_names.append(fname)
        # print(feature_values)
    return feature_names, feature_values


def getall_itd_features(signal):
    ITDFeatures = []
    feature_names = []
    tolerance = 0.20 * np.std(signal)
    # print('calling getShannonEntropy')
    ITDFeatures.append(getShannonEntropy(signal))
    feature_names.append('shannon_entrp')
    '''Not using Shannon impelementation of pyentrp because they dont bin the numbers.. they expect the discritized data'''
    # print('calling getSampleEntropy')
    ITDFeatures.append(getSampleEntropy(signal, DefaultParam.SAMPLE_LENGTH_ITD, tolerance))
    feature_names.append('sample_entrp')
    # print('calling multiscale_entropy')
    temp_vect = getMultiscaleEntropy(signal, DefaultParam.SAMPLE_LENGTH_ITD, maxscale=DefaultParam.MSCALE_ITD)
    ITDFeatures += temp_vect
    for i, v in enumerate(temp_vect):
        feature_names.append('multiscale_entrp'+str(i))

    '''Calculate the Multiscale Entropy of the given time series considering
       different time-scales of the time series.

       Args:
           time_series: Time series for analysis
           sample_length: Bandwidth or group of points
           tolerance: Tolerance (default = 0.1...0.2 * std(time_series))

       Returns:
           Vector containing Multiscale Entropy'''
    # print('calling permutation_entropy')
    ITDFeatures.append(ent.permutation_entropy(signal, DefaultParam.ORDER_OF_PERMUTE_ITD, DefaultParam.DELAY_ITD))
    feature_names.append('permute_entrp')

    '''Calculate the Permutation Entropy
    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay
    Returns:
        Vector containing Permutation Entropy
    '''
    # print('calling multiscale_permutation_entropy!')
    # print(signal.shape)
    # print(ParameterskNN.ORDER_OF_PERMUTE_ITD, ParameterskNN.DELAY_ITD, ParameterskNN.SCALE_ITD)
    temp_vect = ent.multiscale_permutation_entropy(signal, DefaultParam.ORDER_OF_PERMUTE_ITD, DefaultParam.DELAY_ITD, DefaultParam.SCALE_ITD)
    ITDFeatures += temp_vect
    for i, v in enumerate(temp_vect):
        feature_names.append('multiscale_permute_entrp'+str(i))
    '''Calculate the Multiscale Permutation Entropy
        Args:
            time_series: Time series for analysis
            m: Order of permutation entropy
            delay: Time delay
            scale: Scale factor
        Returns:
            Vector containing Multiscale Permutation Entropy
    '''
    # print('calling getCompositeMultiscaleEntropy!')
    temp_vect= getCompositeMultiscaleEntropy(signal, DefaultParam.SAMPLE_LENGTH_ITD, DefaultParam.DELAY_ITD, tolerance)
    ITDFeatures += temp_vect
    for i, v in enumerate(temp_vect):
        feature_names.append('composite_multiscale_entrp'+str(i))
    '''Calculate the Composite Multiscale Entropy of the given time series.
    Args:
        time_series: Time series for analysis
        sample_length: Number of sequential points of the time series
        scale: Scale factor
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing Composite Multiscale Entropy'''

    temp_names, temp_vect= extract_itd_feature(signal)
    ITDFeatures += temp_vect
    feature_names= feature_names+temp_names
    return feature_names, ITDFeatures
