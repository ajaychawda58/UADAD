import os
import pickle
import gc
import sys
import argparse

import torch
from utils import *
import pandas as pd

from torch import optim
from torch.utils.data import DataLoader

from Code.datasets.car_insurance import Car_insurance
from Code.datasets.vehicle_claims import Vehicle_Claims
from Code.datasets.vehicle_insurance import Vehicle_Insurance

from Code.classic_ML.SOM import som_train, som_pred, som_embedding_data
from minisom import MiniSom

#read_inputs
def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vehicle_claims', type=str)
    parser.add_argument('--embedding', dest='embedding', help='True, False', default= False, type=bool)
    parser.add_argument('--encoding', dest='encoding', help='one_hot, label', default= 'label_encode', type=str)
    parser.add_argument('--numerical', dest='numerical', help='True False', default=False, type=bool)
    parser.add_argument('--somsize', dest='somsize', help='10, 20', default = 10, type=int)
    parser.add_argument('--somlr', dest='somlr', help='0.1, 1', default = 1, type=float)
    parser.add_argument('--somsigma', dest='somsigma', help='0.05, 0.1', default = 0.05, type=float)
    parser.add_argument('--somiter', dest='somiter', help='100000', default = 10000, type=int)
    parser.add_argument('--mode', dest='mode', help='train', default='train', type=str)
    parser.add_argument('--threshold', dest='threshold', help=0.5, default=70, type=float)
    args = parser.parse_args()
    return args

args = parse_args()
embedding = args.embedding
encoding = args.encoding
numerical = args.numerical
mode = args.mode
save_path = os.path.join("model", "som" + "_" + str(args.dataset) + "_" + str(args.encoding) +  "_" + str(args.somsize) + "_" + str(embedding))

#read data
# get labels from dataset and drop them if available

if args.dataset == "car_insurance":
    numerical_cols = ['months_as_customer',	'age',	'policy_deductable',	'policy_annual_premium',	'insured_zip'	,'capital-gains',
        	'capital-loss',	'incident_hour_of_the_day',	'number_of_vehicles_involved',	'bodily_injuries',	'witnesses',	'total_claim_amount',
            	'injury_claim',	'property_claim',	'vehicle_claim',	'auto_year']
    if encoding == "label_encode":
        path = 'data/car_insurance/train_label.csv'
        test_path = 'data/car_insurance/test_label.csv'
        label_file = "data/car_insurance/train_label_Y.csv"
        test_label_file = "data/car_insurance/test_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/car_insurance/train_OH.csv'
        label_file = "train_OH_Y.csv"
        test_path = 'data/car_insurance/test_OH.csv'
        test_label_file = 'data/car_insurance/test_OH_Y.csv'
    if encoding == "gel_encode":
        path = 'data/car_insurance/train_gel.csv'
        label_file = "data/car_insurance/train_gel_Y.csv" 
        test_path = 'data/car_insurance/test_gel.csv'
        test_label_file = "data/car_insurance/test_gel_Y.csv"
    if embedding:
        path = 'data/car_insurance/train.csv'
        test_path = 'data/car_insurance/test.csv'
        test_label_file = 'test_Y.csv'
        dataset = Car_insurance(embedding_layer = embedding,  encoding = encoding,  path = path)
        test_dataset = Car_insurance(embedding_layer = embedding,  encoding = encoding,  path = test_path, label_file = test_label_file)
if args.dataset == "vehicle_claims":
    numerical_cols = ['Year', 'WeekOfMonth',	'WeekOfMonthClaimed',	'RepNumber',	'DriverRating', 'Age']
    if encoding == "label_encode":
        path = 'data/vehicle_claims/train_label.csv'
        label_file = "data/vehicle_claims/train_label_Y.csv"
        test_path = 'data/vehicle_claims/test_label.csv'
        test_label_file = "data/vehicle_claims/test_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/vehicle_claims/train_OH.csv'
        label_file = "data/vehicle_claims/train_OH_Y.csv"
        test_path = 'data/vehicle_claims/test_OH.csv'
        test_label_file = "data/vehicle_claims/test_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/vehicle_claims/train_gel.csv'
        label_file = "data/vehicle_claims/train_gel_Y.csv"
        test_path = 'data/vehicle_claims/test_gel.csv'
        test_label_file = "data/vehicle_claims/test_gel_Y.csv"
    if embedding:
        path = 'data/vehicle_claims/train.csv'
        test_path = 'data/vehicle_claims/test.csv'
        test_label_file = 'test_Y.csv'
        dataset = Vehicle_Claims(embedding_layer = embedding,  encoding = encoding,  path = path)
        test_dataset = Vehicle_Claims(embedding_layer = embedding,  encoding = encoding,  path = test_path, label_file = test_label_file)
if args.dataset == "vehicle_insurance":
    numerical_cols = ['Runned_Miles',	'Price',	'repair_cost',	'repair_hours']
    if encoding == "label_encode":
        path = 'data/vehicle_insurance/train_label.csv'
        label_file = "data/vehicle_insurance/train_label_Y.csv"
        test_path = 'data/vehicle_insurance/test_label.csv'
        test_label_file = "data/vehicle_insurance/test_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/vehicle_insurance/train_OH.csv'
        label_file = "data/vehicle_insurance/train_OH_Y.csv"
        test_path = 'data/vehicle_insurance/test_OH.csv'
        test_label_file = "data/vehicle_insurance/test_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/vehicle_insurance/train_gel.csv'
        label_file = "data/vehicle_insurance/train_gel_Y.csv"
        test_path = 'data/vehicle_insurance/test_gel.csv'
        test_label_file = "data/vehicle_insurance/test_gel_Y.csv"
    if embedding:
        path = 'data/vehicle_insurance/train.csv'
        test_path = 'data/vehicle_insurance/test.csv'
        test_label_file = 'test_Y.csv'
        dataset = Vehicle_Insurance(embedding_layer = embedding,  encoding = encoding,  path = path)
        test_dataset = Vehicle_Insurance(embedding_layer = embedding,  encoding = encoding,  path = test_path, label_file = test_label_file)
    
train_data = load_data(path)
test_data = load_data(test_path)



#Select features      
if args.numerical:
    train_data = train_data[numerical_cols]  


print(save_path)

#Convert to torch tensors
if not embedding:
    if args.dataset == 'vehicle_insurance':
        train_data = train_data.drop('PolicyNumber', axis=1)
        test_data = test_data.drop('PolicyNumber', axis=1)
    train_data = torch.tensor(train_data.values.astype(np.float32))
    test_data = torch.tensor(test_data.values.astype(np.float32))
    y_test = load_data(test_label_file) 

    


   

def train():
    if embedding:
        emb = dataset.embedding_sizes
        dataloader = DataLoader(dataset, batch_size = 32)
        som = MiniSom(x= args.somsize, y= args.somsize, input_len=dataset.input_dim, sigma=args.somsigma, learning_rate=args.somlr)
        for i, data in enumerate(dataloader):
            data = som_embedding_data(data[0], data[1], emb)
            data = data.detach().numpy()
            som.train_random(data, args.somiter)
    else:
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        som = MiniSom(x= args.somsize, y= args.somsize, input_len=train_data.shape[1], sigma=args.somsigma, learning_rate=args.somlr)
        for i, data in enumerate(dataloader):
            print(data[0].shape)
            som.train_random(data[0], args.somiter)
    print("SOM training done.")
    with open(save_path, 'wb') as outfile:
        pickle.dump(som, outfile)


def eval():
    with open(save_path, 'rb') as infile:
        som = pickle.load(infile)
    if embedding:
        emb = dataset.embedding_sizes
        q_error = []
        y_test = []
        dataloader = DataLoader(test_dataset, batch_size = 512)
        for i, data in enumerate(dataloader):
            y_test = np.hstack((y_test, data[2]))
            data = som_embedding_data(data[0], data[1], emb)
            data = data.detach()
            error = som_pred(som, data)
            q_error = np.hstack((q_error, error))   
        error_threshold = np.percentile(q_error, (100-args.threshold)/100)
        is_anomaly = q_error > error_threshold
        y_pred = np.multiply(is_anomaly, 1)
        p, r, f, a = get_scores(y_pred, y_test, q_error)
        tn, fp, fn, tp = get_confusion_matrix(y_pred, y_test)
        pd.DataFrame({'Label': y_test, 'Score': q_error}).to_csv((save_path + '.csv'), index=False)

    else:
        y_test = load_data(test_label_file) 
        q_error = som_pred(som, test_data)
        error_threshold = np.percentile(q_error, (100-args.threshold)/100)
        is_anomaly = q_error > error_threshold
        y_pred = np.multiply(is_anomaly, 1)
        p, r, f, a = get_scores(y_pred, y_test, q_error)
        tn, fp, fn, tp = get_confusion_matrix(y_pred, y_test)
        pd.DataFrame({'Label': y_test.iloc[:,0], 'Score': q_error}).to_csv((save_path + '.csv'), index=False)
    print("Precision:", p, "Recall:", r, "F1 Score:", f, "AUROC:", a) 
    print("True Positive:", tp, "False Positive:", fp, "True Negative:", tn, "False Negative:", fn)
    

if __name__ == '__main__':
    if mode == 'train':
        train()
    else:
        eval()
        

