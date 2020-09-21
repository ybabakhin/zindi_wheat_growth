#!/usr/bin/env bash

#python train.py --multirun \
#training.model_id=911 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#data_mode=bad_quality \
#training.max_epochs=20

#python train.py --multirun \
#training.model_id=115 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_911/fold_${training.fold}/'

#python train.py --multirun \
#training.model_id=915 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#data_mode.pseudolabels_path=/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_114/bad_pseudo.csv

#python train.py --multirun \
#training.model_id=917 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#data_mode.pseudolabels_path='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_114/bad_pseudo_fold_${training.fold}.csv'

#python train.py --multirun \
#training.model_id=152 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet152 \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_919/fold_${training.fold}/'

#python train.py --multirun \
#training.model_id=159 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#callbacks.model_checkpoint.save_top_k=2
#
python train.py --multirun \
model.model_id=923 \
model.architecture_name=resnet50 \
training.fold=0,1,2,3,4 \
data_mode=bad_quality \
training.max_epochs=10 \
callbacks.model_checkpoint.save_last=true

python train.py --multirun \
model.model_id=163 \
model.architecture_name=resnet50 \
training.fold=0,1,2,3,4 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_923/fold_${training.fold}/'
