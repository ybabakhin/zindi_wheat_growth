#!/usr/bin/env bash

python train.py --multirun \
training.model_id=13 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet34

python train.py --multirun \
training.model_id=14 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50
