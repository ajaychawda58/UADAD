import os
import argparse
import numpy as np
import pandas as pd

import torch 
from utils import *

from torch.utils.data import DataLoader

from Code.unsupervised_methods.som_dagmm.model import DAGMM, SOM_DAGMM
from Code.unsupervised_methods.som_dagmm.compression_network import CompressionNetwork
from Code.unsupervised_methods.som_dagmm.estimation_network import EstimationNetwork
from Code.unsupervised_methods.som_dagmm.gmm import GMM, Mixture

from Code.unsupervised_methods.rsrae.rsrae import RSRAE

from Code.datasets.car_insurance import Car_insurance
from Code.datasets.vehicle_claims import Vehicle_Claims
from Code.datasets.vehicle_insurance import Vehicle_Insurance

#read_inputs
def parse_args():
    
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vehicle_claims', type=str)
    parser.add_argument('--data', dest='data', help='type of data', default=False, type=bool)
    parser.add_argument('--embedding', dest='embedding', help='True, False', default= False, type=bool)
    parser.add_argument('--encoding', dest='encoding', help='one_hot, label', default= 'label_encode', type=str)
    parser.add_argument('--model', dest='model', help='som, dagmm, somdagmm, rsrae', default='dagmm', type=str)
    parser.add_argument('--numerical', dest='numerical', help='True False', default=False, type=bool)
    parser.add_argument('--batch_size', dest='batch_size', help='32', default = 32, type=int)
    parser.add_argument('--latent_dim', dest='latent_dim', help='1,2', default = 2, type=int)
    parser.add_argument('--num_mixtures', dest='num_mixtures', help='1,2', default = 2, type=int)
    parser.add_argument('--dim_embed', dest='dim_embed', help='1,2', default = 4, type=int)
    parser.add_argument('--rsr_dim', dest='rsr_dim', help='1,2', default = 10, type=int)
    parser.add_argument('--epoch', dest='epoch', help='1', default='1', type=int)
    parser.add_argument('--threshold', dest='threshold', help=0.5, default=0.5, type=float)
    parser.add_argument('--save_path', dest="save_path", help='models/<file_name>', type=str)
    parser.add_argument('--som_save_path', dest="som_save_path", help='models/<file_name>', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
epochs = args.epoch
embedding = args.embedding
encoding = args.encoding
normal = args.data
batch_size = args.batch_size
numerical = args.numerical
save_path = args.save_path




#Test Datasets
if args.dataset == "car_insurance":
    if encoding == "label_encode":
        path = 'data/car_insurance/test_label.csv'
        label_file = "test_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/car_insurance/test_OH.csv'
        label_file = "test_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/car_insurance/test_gel.csv'
        label_file = "test_gel_Y.csv"
    if embedding:
        path = 'data/car_insurance/test.csv'
        label_file = 'test_Y.csv'
    dataset = Car_insurance(embedding_layer = embedding, encoding = encoding, path = path, label_file=label_file)
if args.dataset == "vehicle_claims":
    path = 'data/vehicle_claims/test.csv'
    label_file = 'test_Y.csv'
    if encoding == "label_encode":
        path = 'data/vehicle_claims/test_label.csv'
        label_file = "test_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/vehicle_claims/test_OH.csv'
        label_file = "test_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/vehicle_claims/test_gel.csv'
        label_file = "test_gel_Y.csv"
    if embedding:
        path = 'data/vehicle_claims/test.csv'
        label_file = 'test_Y.csv'
    dataset = Vehicle_Claims(embedding_layer = embedding, encoding = encoding, path = path, label_file=label_file)
if args.dataset == "vehicle_insurance":
    if encoding == "label_encode":
        path = 'data/vehicle_insurance/test_label.csv'
        label_file = "test_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/vehicle_insurance/test_OH.csv'
        label_file = "test_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/vehicle_insurance/test_gel.csv'
        label_file = "test_gel_Y.csv"
    if embedding:
        path = 'data/vehicle_insurance/test.csv'
        label_file = 'test_Y.csv'
    dataset = Vehicle_Insurance(embedding_layer = embedding, encoding = encoding, path = path, label_file=label_file)

#Parameters for embedding layer initialization and numerical features
emb = None
if embedding:
    emb = dataset.embedding_sizes        
input_dim = dataset.input_dim
output_dim = dataset.output_dim
if numerical:
    input_dim = output_dim = dataset.cont_cols.shape[1]

#DataLoader
dataloader = DataLoader(dataset, batch_size= batch_size)
score = []
label = []

if args.model == "dagmm": 
    from Code.unsupervised_methods.dagmm_self.model import DAGMM
    from Code.unsupervised_methods.dagmm_self.compression_network import CompressionNetwork
    from Code.unsupervised_methods.dagmm_self.estimation_network import EstimationNetwork
    from Code.unsupervised_methods.dagmm_self.gmm import GMM, Mixture
    compression = CompressionNetwork(embedding, numerical, input_dim, output_dim, emb, args.latent_dim)
    estimation = EstimationNetwork(args.dim_embed, args.num_mixtures)
    gmm = GMM(args.num_mixtures,args.dim_embed)
    mix = Mixture(args.dim_embed)
    net = DAGMM(compression, estimation, gmm)
    path = os.path.split(save_path)
    #net.load_state_dict(torch.load(save_path), strict=False)
    net = torch.load(save_path)
    net.eval()
    for i, data in enumerate(dataloader):
        rec_data = torch.cat([data[0], data[1]], -1)
        if numerical:
            rec_data = data[1]
        out = net(data[0], data[1], rec_data)
        out = out.detach().numpy().reshape(-1)
        L =  data[2].detach().numpy().reshape(-1)
        score = np.hstack((score, out))   
        label = np.hstack((label, L))
    threshold = np.percentile(score, (100 - args.threshold), axis=0)  
    y_pred = (score < threshold).astype(int)
    y_test = label
if args.model == "somdagmm":
    compression = CompressionNetwork(embedding, numerical, input_dim, output_dim, emb, args.latent_dim)
    estimation = EstimationNetwork(args.dim_embed, args.num_mixtures)
    gmm = GMM(args.num_mixtures,args.dim_embed)
    mix = Mixture(args.dim_embed)
    dagmm = DAGMM(compression, estimation, gmm)
    net = SOM_DAGMM(dagmm, embedding, numerical, emb)
    net = torch.load(save_path)
    net.eval()

    for i, data in enumerate(dataloader):
        rec_data = torch.cat([data[0], data[1]], -1)
        if numerical:
            rec_data = data[1]
        out = net(data[0], data[1], rec_data, args.som_save_path)
        out = out.detach().numpy().reshape(-1)
        L =  data[2].detach().numpy().reshape(-1)
        score = np.hstack((score, out))   
        label = np.hstack((label, L))
    threshold = np.percentile(score, (100 - args.threshold), axis=0)  
    print(threshold)
    y_pred = (score < threshold).astype(int)
    y_test = label
if args.model == "rsrae":
    net = RSRAE(embedding, numerical, input_dim, output_dim, emb, args.rsr_dim, args.latent_dim)
    net.load_state_dict(torch.load(save_path))
    #net = torch.load(save_path)
    net.eval()
    for i, data in enumerate(dataloader):
        enc, dec, latent, A = net(data[0], data[1])
        rec_data = torch.cat([data[0], data[1]], -1)
        if numerical:
            rec_data = data[1]
        out = relative_euclidean_distance(rec_data, dec)
        out = out.detach().numpy().reshape(-1)
        L =  data[2].detach().numpy().reshape(-1)
        score = np.hstack((score, out))   
        label = np.hstack((label, L))   
    threshold = np.percentile(score, args.threshold, axis=0)
    y_pred = (score > threshold).astype(int)
    y_test = label  
    #print(y_pred, score)
# Precision, Recall, F1
pd.DataFrame({'Score':score, 'Label':y_test}).to_csv((save_path + '.csv'), index=False)
p, r, f, a = get_scores(y_pred, y_test, score)
tn, fp, fn, tp = get_confusion_matrix(y_pred, y_test)
print("Precision:", p, "Recall:", r, "F1 Score:", f, "AUROC:", a)
print("True Positive:", tp, "False Positive:", fp, "True Negative:", tn, "False Negative:", fn)


