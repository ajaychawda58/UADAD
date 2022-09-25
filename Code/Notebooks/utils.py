import pandas as pd
import os
import sys
import numpy as np
import qgel
from math import floor

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def load_data(file_path, names = None):
    names = names
    data = pd.read_csv(file_path, names = names)
    return data

def one_hot_encoding(data, cols):
    encode_data = data[cols]
    encoded_data = pd.DataFrame()
    for column in encode_data:
        new_data = pd.get_dummies(encode_data[column], prefix=column)
        encoded_data = pd.concat([encoded_data, new_data], axis=1)
    data.drop(cols, axis=1, inplace=True)
    data = pd.concat([data, encoded_data], axis=1)
    return data

def label_encoding(data, cols):
    encode_data = data[cols]
    encoded_data = pd.DataFrame()
    encoder = preprocessing.LabelEncoder()
    for column in encode_data:
        new_data = encoder.fit_transform(encode_data[column])
        new_data = pd.DataFrame(new_data, columns=[column])
        encoded_data = pd.concat([encoded_data, new_data], axis=1)
    data.drop(cols, axis=1, inplace=True)
    data = pd.concat([data, encoded_data], axis=1)
    return data

def gel_encoding(data, cols):
    encode_data = data[cols]
    encoded_data = pd.DataFrame()
    for column in encode_data:
        new_data = pd.get_dummies(encode_data[column], prefix=column)
        encoded_data = pd.concat([encoded_data, new_data], axis=1)
        embedding, vectors, source_data_ = qgel.qgel(one_hot_data = encoded_data, 
                                                 k = int(len(cols)), 
                                                 learning_method = 'unsupervised') 
    data.drop(cols, axis=1, inplace=True)
    data = pd.concat([data, pd.DataFrame(embedding, columns=[x for x in range(0,int(len(cols)))])], axis=1) 
    return data
    
def get_labels(data, name):
    if name == 'credit_card':
        label = data['Class']
        data.drop(['Class'], axis = 1, inplace=True)        
    if name == 'arrhythmia':
        label = data['class']
        anomaly_classes = [3,4,5,7,8,9,10,14,15]
        label = [1 if i in anomaly_classes else 0 for i in label]
        label = np.array(label)  
    if name == 'kdd':
        label = data[41]
        label = np.where(label == "normal", 0, 1)
        data.drop([41], axis = 1, inplace=True) 
    if name == 'vehicle_claims':
        label = data['Label']
        data.drop(['Label'], axis = 1, inplace=True)
    if name == 'vehicle_insurance':
        label = data['FraudFound_P']
        data.drop(['FraudFound_P'], axis = 1, inplace=True)
    if name == 'car_insurance':
        label = data['fraud_reported']
        label = np.where(label == "Y", 1, 0)
        label = pd.DataFrame(label, columns=["Label"])
        data.drop(['fraud_reported'], axis = 1, inplace=True)       
    return label

def get_normal_data(data, name):
    sample_len = len(data)/2
    if name == 'credit_card':
        label = data['Class']
        ind = np.where(data['Class'] == 1)       
    if name == 'arrhythmia':
        label = data['class']
        anomaly_classes = [3,4,5,7,8,9,10,14,15]
        label = [1 if i in anomaly_classes else 0 for i in label]
        label = np.array(label)      
    if name == 'kdd':
        label = data[41]
        label = np.where(label == "normal", 0, 1)    
    if name == 'vehicle_claims':
        label = data['Label']    
    if name == 'vehicle_insurance':
        label = data['FraudFound_P']    
    if name == 'car_insurance':
        label = data['fraud_reported']
        label = np.where(label == "Y", 1, 0)
    data['label'] = label
    normal_data = data[data['label'] == 0]
    sampled_data = normal_data.sample(n= int(sample_len))
    sampled_data = sampled_data.drop('label', axis = 1)
    return sampled_data

def get_scores(y_pred, y):
    precision = precision_score(y, y_pred, labels = [0], average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    auc = roc_auc_score(y, y_pred, average="weighted")
    return precision, recall, f1, auc

def get_confusion_matrix(y_pred, y):
    tn, fp, fn, tp = confusion_matrix(y_pred, y).ravel()
    return tn, fp, fn, tp

def normalize_cols(data):
    sc = MinMaxScaler (feature_range = (0,1))
    data = sc.fit_transform(data)
    data = pd.DataFrame(data)  
    return data  

def merge_cols(data_1, data_2):
    data = pd.concat([data_1, data_2], axis=1)
    return data

def remove_cols(data, cols):
    data.drop(cols, axis=1, inplace=True)
    return data

def split_data(data, Y, split=0.7):
    train = data.loc[:split*len(data),]
    test = data.loc[split*len(data)+1:,]
    Y_train = Y[:floor(split*len(data)),]
    Y_test = Y[floor(split*len(data))+1:,]
    return train, test, Y_train, Y_test

def fill_na(data):
    data = data.fillna(value = 0, axis=1)
    return data

def relative_euclidean_distance(x1, x2, eps=1e-8):
    num = np.norm(x1 - x2, p=2, dim=1)  
    denom = np.norm(x1, p=2, dim=1)  
    return num / np.max(denom, eps)


def cosine_similarity(x1, x2, eps=1e-8):
    dot_prod = np.sum(x1 * x2, dim=1)  
    dist_x1 = np.norm(x1, p=2, dim=1)  
    dist_x2 = np.norm(x2, p=2, dim=1)  
    return dot_prod / np.max(dist_x1*dist_x2, eps)