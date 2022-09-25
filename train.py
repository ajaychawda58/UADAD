import os
import pickle
import gc
import sys
import argparse

import torch
from Code.datasets.vehicle_claims import Vehicle_Claims 
from utils import *

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Code.unsupervised_methods.rsrae.rsrae import RSRLoss
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
    parser.add_argument('--file_name', dest='file_name', help='model_data_embed_encode_latent_dim_parameters', type=str)
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
som_save_path = args.som_save_path
save_path = os.path.join("model", args.file_name )

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Auditing Datasets
if args.dataset == "car_insurance":
    if encoding == "label_encode":
        path = 'data/car_insurance/train_label.csv'
        label_file = "train_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/car_insurance/train_OH.csv'
        label_file = "train_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/car_insurance/train_gel.csv'
        label_file = "train_gel_Y.csv"
    if embedding:
        path = 'data/car_insurance/train.csv'
    dataset = Car_insurance(embedding_layer = embedding, path = path, label_file=label_file)
if args.dataset == "vehicle_claims":
    if encoding == "label_encode":
        path = 'data/vehicle_claims/train_label.csv'
        label_file = "train_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/vehicle_claims/train_OH.csv'
        label_file = "train_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/vehicle_claims/train_gel.csv'
        label_file = "train_gel_Y.csv"
    if embedding:
        path = 'data/vehicle_claims/train.csv'
    dataset = Vehicle_Claims(embedding_layer = embedding, encoding = encoding, path = path, label_file=label_file)
if args.dataset == "vehicle_insurance":
    if encoding == "label_encode":
        path = 'data/vehicle_insurance/train_label.csv'
        label_file = "train_label_Y.csv"
    if encoding == "one_hot":
        path = 'data/vehicle_insurance/train_OH.csv'
        label_file = "train_OH_Y.csv"
    if encoding == "gel_encode":
        path = 'data/vehicle_insurance/train_gel.csv'
        label_file = "train_gel_Y.csv"
    if embedding:
        path = 'data/vehicle_insurance/train.csv'
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

#Training Models
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
    optimizer =  optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0
        for i, data in enumerate(dataloader):
            rec_data = torch.cat([data[0], data[1]], -1)
            if numerical:
                rec_data = data[1]
            out = net(data[0], data[1], rec_data)
            optimizer.zero_grad()
            L_loss = compression.reconstruction_loss(data[0], data[1], rec_data)
            G_loss = mix.gmm_loss(out=out, L1=1, L2=0.05)
            print(L_loss, G_loss)
            loss = (L_loss + G_loss)/ len(data[1])
            loss.backward()            
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss, epoch)
        print(running_loss)
    writer.flush()
    torch.save(net, save_path)
if args.model == "somdagmm":
    from Code.unsupervised_methods.som_dagmm.model import DAGMM, SOM_DAGMM
    from Code.unsupervised_methods.som_dagmm.compression_network import CompressionNetwork
    from Code.unsupervised_methods.som_dagmm.estimation_network import EstimationNetwork
    from Code.unsupervised_methods.som_dagmm.gmm import GMM, Mixture
    compression = CompressionNetwork(embedding, numerical, input_dim, output_dim, emb, args.latent_dim)
    estimation = EstimationNetwork(args.dim_embed, args.num_mixtures)
    gmm = GMM(args.num_mixtures,args.dim_embed)
    mix = Mixture(args.dim_embed)
    dagmm = DAGMM(compression, estimation, gmm)
    net = SOM_DAGMM(dagmm, embedding, numerical, emb)
    optimizer =  optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0
        for i, data in enumerate(dataloader):
            rec_data = torch.cat([data[0], data[1]], -1)
            if numerical:
                rec_data = data[1]
            out = net(data[0], data[1], rec_data, som_save_path)
            optimizer.zero_grad()
            L_loss = compression.reconstruction_loss(data[0], data[1], rec_data)
            G_loss = mix.gmm_loss(out=out, L1=1, L2=0.05)
            loss = (L_loss + G_loss)/ len(data[1])
            print(L_loss, G_loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar("Loss/train", running_loss, epoch)
        print(running_loss)
    writer.flush()
    torch.save(net, save_path)
if args.model == "rsrae":
    rsr = RSRLoss(0.1,0.1, args.rsr_dim, args.latent_dim).to(device)
    net = RSRAE(embedding, numerical, input_dim, output_dim, emb, args.rsr_dim, args.latent_dim)
    net.to(device)
    optimizer =  optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0
        for i, data in enumerate(dataloader):            
            optimizer.zero_grad()
            enc, dec, latent, A = net(data[0].to(device), data[1].to(device))
            rec_data = torch.cat([data[0], data[1]], -1).to(device)
            if numerical:
                rec_data = data[1]
            rec_loss = net.L21(dec,rec_data).to(device)
            rsr_loss = rsr(enc, A).to(device)
            loss = (rec_loss + rsr_loss)/ len(data[1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss, epoch)
        print(running_loss) 
    writer.flush()
    torch.save(net.state_dict(), save_path)


