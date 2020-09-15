#!/usr/bin/env bash

#python train.py --multirun \
#training.model_id=93 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_907/fold_${training.fold}/'

python train.py --multirun \
training.model_id=109 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_907/fold_${training.fold}/' \
training.augmentations=base_v1
