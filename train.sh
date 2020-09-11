#!/usr/bin/env bash

python train.py --multirun \
training.model_id=49 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.max_epochs=20

python train.py --multirun \
training.model_id=50 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.max_epochs=10

python train.py --multirun \
training.model_id=51 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.max_epochs=40

python train.py --multirun \
training.model_id=52 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.lr=1e-5

python train.py --multirun \
training.model_id=53 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.lr=1e-4
