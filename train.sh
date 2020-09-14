#!/usr/bin/env bash

#python train.py --multirun \
#training.model_id=903 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#data_mode=bad_quality \
#callbacks.model_checkpoint.filepath='${training.logs_dir}model_${training.model_id}/fold_${training.fold}/best' \
#training.max_epochs=20

#python train.py --multirun \
#training.model_id=73 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#training.pretrain_path='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_903/fold_${training.fold}/best.ckpt'

#python train.py --multirun \
#training.model_id=78 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50 \
#training.pretrain_path='/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_903/fold_${training.fold}/best.ckpt' \
#training.regression=true \
#data_mode.num_classes=1
