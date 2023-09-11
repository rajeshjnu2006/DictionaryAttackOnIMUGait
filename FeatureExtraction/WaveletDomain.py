import numpy as np
import pandas as pd
import pywt

from FeatureExtraction import ITheoryDomain

def wavelet_features(signal, motherWavelet, level):
    coeffs = pywt.wavedec(signal, motherWavelet, level=level)
    WaveletFeatures = []
    feature_names = []
    id = 0
    for i, row in enumerate(coeffs):
        for ele in row:
            WaveletFeatures.append(ele)
            feature_names.append('wave'+str(id))
            id =id+1
    return feature_names, WaveletFeatures
