import numpy as np
import pandas as pd
from utils import *
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

def chi_square_test(data, Y, k = 5):
    fs = SelectKBest(chi2, k = k)
    data = fs.fit_transform(data, Y)
    return data   
    
def information_gain(data, Y, k = 5):
    fs = SelectKBest(mutual_info_classif, k=5)
    data = fs.fit_transform(data, Y)
    return data      
    
def sfs (data, Y, k = 5):
    
    
