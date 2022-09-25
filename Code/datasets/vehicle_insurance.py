import os
import pandas as pd
from torch.utils.data import Dataset
import torch 
import numpy as np
from utils import *

class Vehicle_Insurance(Dataset):
    def __init__(self, path, encoding = "label_encode", embedding_layer = False, label_file="train_Y.csv"):
        path = path
        categorical_cols = ['Make', 'AccidentArea',	'Sex',	'MaritalStatus',	'Fault', 'PolicyType',
                    	'VehicleCategory',	'Deductible',	'Days_Policy_Accident',	'Days_Policy_Claim',
            	        'AgeOfVehicle',  'AgeOfPolicyHolder', 'PoliceReportFiled',	'WitnessPresent',
                	   'AgentType',	'NumberOfSuppliments',	'AddressChange_Claim', 'VehiclePrice',
                       'PastNumberOfClaims', 'NumberOfCars', 'BasePolicy', 'Month', 'MonthClaimed',
                       'DayOfWeek', 'DayOfWeekClaimed']
        numerical_cols = ['Year', 'WeekOfMonth',	'WeekOfMonthClaimed',	'RepNumber',	'DriverRating', 'Age']
        data = load_data(path)
        data = remove_cols(data, ['PolicyNumber'])
        label = load_data(os.path.join(os.path.split(path)[0], label_file))
        self.label = label["FraudFound_P"]
        self.cont_cols = data[numerical_cols]
        self.cat_cols = data.drop(numerical_cols, axis=1)
        self.input_dim = self.cat_cols.shape[1] + self.cont_cols.shape[1]
        self.output_dim = self.input_dim
        self.embed = embedding_layer
        if embedding_layer:
            embed_data = load_data(os.path.join(os.path.split(path)[0], "fraud_oracle.csv"))
            embed_data = remove_cols(embed_data, ['FraudFound_P', 'PolicyNumber'])
            cat_cols = embed_data.drop(numerical_cols, axis=1).astype("category")
            embedded_cols = {n: len(col.cat.categories) for n,col in cat_cols.items() if (col.dtype == "category")}
            self.embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
            embedded_col_names = embedded_cols.keys()
            embed = []
            for i, name in enumerate(embedded_col_names):
                embed_elem = {cat : n for n, cat in enumerate(cat_cols[name].cat.categories)}
                embed.append(embed_elem)
                self.cat_cols[name] = self.cat_cols[name].replace(embed_elem)
            self.input_dim = sum(i[1] for i in self.embedding_sizes) + (self.cont_cols).shape[1] 
            self.output_dim = data.shape[1] 
       
    def __len__(self):
        return(len(self.label))
    
    def __getitem__(self, idx):
        self.cont_cols = normalize_cols(self.cont_cols) 
        cat_cols = (self.cat_cols.values.astype(np.float32))
        if self.embed:
            cat_cols = (self.cat_cols.values.astype(np.int32))
        cont_cols = (self.cont_cols.values.astype(np.float32))
        label = (self.label.astype(np.int32))
        return (cat_cols[idx], cont_cols[idx], label[idx])
