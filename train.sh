#!/usr/bin/env bash

#python train.py --multirun \
#training.model_id=26 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet34
#
#python train.py --multirun \
#training.model_id=27 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=resnet50

python train.py --multirun \
training.model_id=30 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50

#python train.py --multirun \
#training.model_id=28 \
#training.fold=0,1,2,3,4 \
#training.architecture_name=efficientnet-b3
