#!/usr/bin/env bash

python train.py --multirun \
training.model_id=35 \
training.fold=0 \
training.architecture_name=resnet50 \
training.num_classes=5 \
training.label_quality=1
