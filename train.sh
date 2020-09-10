#!/usr/bin/env bash

python train.py --multirun \
training.model_id=38 \
training.fold=0,1,2,3,4 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=2
