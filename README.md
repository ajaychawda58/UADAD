# Deep unsupervised anomaly detection with auditing data


## Datasets

1) Car Insurance - Kaggle(https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)
1) Insurance Claim Classification - Kaggle(https://www.kaggle.com/competitions/ml-classification1-2020/data?select=train.csv)
1) Vehicle Insurance - Github(https://github.com/AnalyticsandDataOracleUserCommunity/MachineLearning)
1) Vehicle Claim - Synthetic dataset generated for this thesis.

## Training

### DAGMM/SOM-DAGMM/RSRAE

`python train.py [-h] [--dataset DATASET] [--data DATA] [--embedding EMBEDDING] [--encoding ENCODING] [--model MODEL] [--numerical NUMERICAL]
 [--batch_size BATCH_SIZE] [--latent_dim LATENT_DIM] [--num_mixtures NUM_MIXTURES] [--dim_embed DIM_EMBED] [--rsr_dim RSR_DIM] [--epoch EPOCH]`

- `dataset` - Dataset for training ('vehicle_claims', 'car_insurance', 'vehicle_insurance')
- `data` - Only Normal data or Mixed data (True = Normal data)
- `embedding` - Embedding layer if needed (DEFAULT = False)
- `encoding` - Categorical features encodings (DEFAULT = 'label_encode' | 'one_hot', 'gel_encode')
- `numerical` - Only numerical features if TRUE (DEFAULT = FALSE)
- `batch_size` - (DEFAULT = 32)
- `epoch` - (DEFAULT = 1)
- `latent_dim` - Dimension of latent space in autoencoder (DEFAULT = 2)

**DAGMM**

- `num_mixtures` - Number of gaussian mixture models (DEFAULT =  2)
- `dim_embed` - Dimension of input to estimation network (DEFAULT = 4 | General case = [latent_dim + 2])

**RSRAE**

- `rsr_dim` - Dimension of RSR layer (DEFAULT = 10 | Should be less than latent_dim)

## Evaluation (DAGMM/SOM-DAGMM/RSRAE)

`python eval.py [-h] [--dataset DATASET] [--data DATA] [--embedding EMBEDDING] [--encoding ENCODING] [--model MODEL] [--numerical NUMERICAL] 
[--batch_size BATCH_SIZE] [--latent_dim LATENT_DIM] [--num_mixtures NUM_MIXTURES] [--dim_embed DIM_EMBED] [--rsr_dim RSR_DIM] [--epoch EPOCH] 
[--threshold THRESHOLD]`

### SOM 

`train_som.py [-h] [--dataset DATASET] [--embedding EMBEDDING] [--encoding ENCODING] [--numerical NUMERICAL] [--somsize SOMSIZE] 
[--somlr SOMLR] [--somsigma SOMSIGMA] [--somiter SOMITER] [--mode MODE] [--threshold THRESHOLD]`

- `somsize` - Size of Self Organizing Map
- `somlr` - Learning Rate 
- `somsigma` - Sigma for neighbourhood function
- `somiter` - Number of iterations of SOM
- `mode` - train or eval (DEFAULT = 'train')
- `threshold` - (DEFAULT = 50 | Only in eval mode)

## References

1) https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets
1) https://github.com/GuansongPang/SOTA-Deep-Anomaly-Detection
1) https://deepvisualmarketing.github.io/
1) https://github.com/zhuyitan/IGTD
