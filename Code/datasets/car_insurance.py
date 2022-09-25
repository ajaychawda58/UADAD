import os
import pandas as pd
from torch.utils.data import Dataset
import torch 
import numpy as np
from utils import *

class Car_insurance(Dataset):
    def __init__(self, path, encoding = "label_encode", embedding_layer = False, label_file="train_Y.csv"):
        path = path
        categorical_cols = ['policy_state', 'umbrella_limit', 'insured_sex', 'insured_education_level',
    	'insured_occupation', 'insured_hobbies', 'insured_relationship', 'incident_type',
        'collision_type', 'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city',	
        'property_damage', 'police_report_available', 'auto_make', 'auto_model']
        cols_to_remove = ['policy_number', 'policy_bind_date', 'policy_csl', 'incident_location', 'incident_date', '_c39']
        numerical_cols = ['months_as_customer',	'age',	'policy_deductable',	'policy_annual_premium',	'insured_zip'	,'capital-gains',
        	'capital-loss',	'incident_hour_of_the_day',	'number_of_vehicles_involved',	'bodily_injuries',	'witnesses',	'total_claim_amount',
            	'injury_claim',	'property_claim',	'vehicle_claim',	'auto_year']

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
            embed_data = load_data(os.path.join(os.path.split(path)[0], "insurance_claims.csv"))
            cat_cols = embed_data[categorical_cols].astype("category")
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