#!/usr/bin/env bash

# Inference ***********************
python test.py \
model.model_id=2 \
model.architecture_name=resnet50 \
testing.mode=test \
model.input_size=[512,512]

python test.py --multirun \
model.model_id=3,5 \
model.architecture_name=resnet101 \
testing.mode=test

python test.py \
model.model_id=4 \
model.architecture_name=resnet50 \
testing.mode=test

python test.py \
model.model_id=6 \
model.architecture_name=resnext50_32x4d \
testing.mode=test

# Ensemble
python ensemble.py \
ensemble.model_ids=[2,3,4,5,6] \
testing.mode=test
