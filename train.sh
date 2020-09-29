#!/usr/bin/env bash

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
#model.model_id=172 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=30 \
#scheduler=plateau

#python train.py --multirun \
#model.model_id=923 \
#model.architecture_name=resnet50 \
#training.fold=0,1,2,3,4 \
#data_mode=bad_quality \
#training.max_epochs=10 \
#callbacks.model_checkpoint.save_last=true \
#scheduler=plateau
#
#python train.py --multirun \
#model.model_id=166 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_923/fold_${training.fold}/'

#python train.py --multirun \
#model.model_id=174 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=20 \
#scheduler=plateau \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_166/fold_${training.fold}/' \
#training.batch_size=24

#python train.py --multirun \
#model.model_id=179 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_923/fold_${training.fold}/' \
#model.regression=true \
#data_mode.num_classes=1
#
#python train.py --multirun \
#model.model_id=925 \
#model.architecture_name=resnet101 \
#training.fold=0,1,2,3,4 \
#data_mode=bad_quality \
#training.max_epochs=10 \
#callbacks.model_checkpoint.save_last=true \
#scheduler=plateau
#
#python train.py --multirun \
#model.model_id=177 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet101 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_925/fold_${training.fold}/'

#python test.py --multirun \
#model.model_id=166 \
#model.architecture_name=resnet50 \
#testing.evaluate=false \
#testing.pseudolabels=true \
#testing.folds=[0,1,2,3,4]
#
#python test.py --multirun \
#model.model_id=166 \
#model.architecture_name=resnet50 \
#testing.evaluate=false \
#testing.pseudolabels=true \
#testing.folds=[0],[1],[2],[3],[4]

#python train.py --multirun \
#model.model_id=942 \
#model.architecture_name=resnet50 \
#training.fold=0,1,2,3,4 \
#data_mode=bad_quality \
#training.max_epochs=10 \
#callbacks.model_checkpoint.save_last=true \
#scheduler=plateau \
#training.batch_size=24
#
#python train.py --multirun \
#model.model_id=205 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_942/fold_${training.fold}/' \
#training.batch_size=24

#python train.py --multirun \
#model.model_id=944 \
#model.architecture_name=resnet101 \
#training.fold=0,1,2,3,4 \
#data_mode=bad_quality \
#training.max_epochs=10 \
#callbacks.model_checkpoint.save_last=true \
#scheduler=plateau \
#training.label_augmentation=0.2
#
#python train.py --multirun \
#model.model_id=211 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet101 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_944/fold_${training.fold}/' \
#training.label_augmentation=0.1
#
#python train.py --multirun \
#model.model_id=215 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=20 \
#scheduler=plateau \
#training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_210/fold_${training.fold}/' \
#training.batch_size=24 \
#model.input_size=[512,512]

python train.py --multirun \
model.model_id=216 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet101 \
training.max_epochs=20 \
scheduler=plateau \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_211/fold_${training.fold}/' \
training.batch_size=16 \
model.input_size=[512,512]

python train.py --multirun \
model.model_id=217 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnext50_32x4d \
training.max_epochs=20 \
scheduler=plateau \
training.pretrain_dir='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_214/fold_${training.fold}/' \
training.batch_size=20 \
model.input_size=[512,512]
