#!/usr/bin/env bash

python train.py --multirun \
training.model_id=40 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet34 \
training.num_classes=5 \
training.label_quality=2

python train.py --multirun \
training.model_id=41 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2

python train.py --multirun \
training.model_id=42 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.input_size=512 \
training.train_batch_size=16 \
training.valid_batch_size=16

python train.py --multirun \
training.model_id=43 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.augmentations=v2

python train.py --multirun \
training.model_id=44 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2 \
training.augmentations=v2 \
training.input_size=512 \
training.train_batch_size=16 \
training.valid_batch_size=16
