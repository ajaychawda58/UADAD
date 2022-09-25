import os
import pandas as pd
from torch.utils.data import Dataset
import torch 
import numpy as np
from utils import *

class Vehicle_Claims(Dataset):
    def __init__(self, path, encoding = "label_encode", embedding_layer = False, label_file="train_Y.csv"):
        path = path
        categorical_cols = ['Maker','Reg_year', ' Genmodel', 'Color', 'Bodytype', 'Engin_size', 'Gearbox', 'Fuel_type',
                        'Seat_num', 'Door_num', 'issue', 'issue_id', 'repair_complexity']
        cols_to_remove = [' Genmodel_ID', 'Adv_year', 'Adv_month', 'Adv_day', 'breakdown_date', 'repair_date', 'category_anomaly'] 
        numerical_cols = ['Runned_Miles',	'Price',	'repair_cost',	'repair_hours']
        data = load_data(path)
        #data = remove_cols(data, cols_to_remove)
        label = load_data(os.path.join(os.path.split(path)[0], label_file))
        self.label = label["Label"]
        self.cont_cols = data[numerical_cols]
        self.cat_cols = data.drop(numerical_cols, axis=1)
        self.input_dim = self.cat_cols.shape[1] + self.cont_cols.shape[1]
        self.output_dim = self.input_dim
        self.embed = embedding_layer
        if embedding_layer:
            embed_data = load_data(os.path.join(os.path.split(path)[0], "vehicle_claims_labeled.csv"))
            embed_data = remove_cols(embed_data, cols_to_remove)
            embed_data = remove_cols(embed_data, ['Label'])
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




