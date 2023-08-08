# UMLGA
Submission to WSDM 2024: Unsupervised Graph Meta Learning via Local Subgraph Augmentation

## Enviorment and dependency
Ubuntu 20.04

python3.8

dgl-cu102==0.5.1

torch==1.12.1+cu116

torch-cluster==1.5.7

torch-scatter==2.0.5

scikit-learn==1.2.1

scipy==1.10.0

ogb==1.3.5

openne==2.0

## How to run ?

```
python3 main_augmentation.py --dataset $DATASET --way $WAY --shot $SHOT --augmentation_method $METHOD --augmentation_parameter $PARAMETER
```
