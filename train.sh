#!/usr/bin/env bash

python train.py --multirun \
training.model_id=908 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
data_mode=bad_quality \
training.max_epochs=20 \
training.input_size=512 \
training.batch_size=24

python train.py --multirun \
training.model_id=86 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_907/fold_${training.fold}/' \
training.input_size=512 \
training.batch_size=24

python train.py --multirun \
training.model_id=87 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_907/fold_${training.fold}/' \
training.regression=true \
data_mode.num_classes=1 \
training.input_size=512 \
training.batch_size=24

python train.py --multirun \
training.model_id=88 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_908/fold_${training.fold}/' \
training.input_size=512 \
training.batch_size=24

python train.py --multirun \
training.model_id=89 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_84/fold_${training.fold}/' \
training.input_size=512 \
training.batch_size=24

python train.py --multirun \
training.model_id=90 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_84/fold_${training.fold}/'
