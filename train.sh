#!/usr/bin/env bash

#python train.py --multirun \
#training.model_id=911 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#data_mode=bad_quality \
#training.max_epochs=20
#
#python train.py --multirun \
#training.model_id=115 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_911/fold_${training.fold}/'

python train.py --multirun \
training.model_id=920 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
data_mode=bad_quality \
training.max_epochs=20 \
callbacks.model_checkpoint.save_last=true

python train.py --multirun \
training.model_id=161 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_920/fold_${training.fold}/'

