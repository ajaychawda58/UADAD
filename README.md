# Unsupervised Anomaly detection for Auditing Data and Impact of Cetgorical Encodings


## Datasets

1) Vehicle Claim - Synthetic dataset created using DVI dataset.
1) Car Insurance - Kaggle(https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)
1) Vehicle Insurance - Github(https://github.com/AnalyticsandDataOracleUserCommunity/MachineLearning)

## Vehicle Claim dataset

The code to create dataset is available [here](https://github.com/ajaychawda58/UADAD/blob/main/Code/Notebooks/create_dataset.ipynb).

- `Maker` - *Categorical* - The brand of the vehicle.
- `GenModel` - *Categorical* - The model of the vehicle.
- `Color` - *Categorical* - Colour of the vehicle.
- `Reg_Year` - *Categorical* - Year of Registration.
- `Body_Type` - *Categorical* - Eg. SUV, Convertible.
- `Runned_Miles` - *Numerical* - Distance covered by the vehicle.
- `Engin_Size` - *Categorical* - Size of engine.
- `GearBox` - *Categorical* - Automatic, Manual.
- `FuelType` - *Categorical* - Petrol, Diesel.
- `Price` -  *Numerical* - Price of vehicle.
- `Seat_num` - *Numerical* - Number of seats.
- `Door_num` -  *Numerical* - Number of Doors.
- `issue` - *Categorical* - Type of damage.
- `issue_id` - *Categorical* - Specific damage.
- `repair_complexity` - *Categorical* - Difficulty to repair the vehicle.
- `repair_hours` -  *Numerical* - Time required to finish the job.
- `repair_cost` - *Numerical* - Cost of repair.

Other attributes are not used for evaluation in this work. 
`breakdown_date` and `repair_date` were added with the idea of inserting anomalies based on the number of days required to repair the vehicle.


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

1) DVI dataset - https://deepvisualmarketing.github.io/
1) RSRAE - https://github.com/marrrcin/rsrlayer-pytorch
1) DAGMM - https://github.com/RomainSabathe/dagmm
1) SOM - https://github.com/JustGlowing/minisom
1) NeuTraL-AD - https://github.com/boschresearch/NeuTraL-AD
1) LOE - https://github.com/boschresearch/LatentOE-AD


Please consider citing our work if you found this repository to be helpful.
```
@article{
    Author = {Ajay Chawda and Axel Vierling and Karsten Berns},
    Title = {Unsupervised Anomaly detection for Auditing Data and Impact of Cetgorical Encodings},
    Journal = {https://arxiv.org/abs/2210.14056},
    Year = {2022},
}
```
