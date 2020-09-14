#!/usr/bin/env bash

python train.py --multirun \
training.model_id=907 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
data_mode=bad_quality \
training.max_epochs=20

python train.py --multirun \
training.model_id=84 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_907/fold_${training.fold}/'

python train.py --multirun \
training.model_id=85 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_907/fold_${training.fold}/' \
training.regression=true \
data_mode.num_classes=1
