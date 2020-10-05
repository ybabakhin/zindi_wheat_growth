#!/usr/bin/env bash

# Prepare the data
python src/create_folds.py
# ***************************************

# Pretrain on bad labels ***********************
python train.py --multirun \
model.model_id=101 \
model.architecture_name=resnet50 \
training.fold=0,1,2,3,4 \
data_mode=bad_quality \
training.max_epochs=10 \
callbacks.model_checkpoint.save_last=true \
scheduler=plateau

# Finetune on good labels
python train.py --multirun \
model.model_id=1 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet50 \
training.max_epochs=50 \
scheduler=plateau \
training.pretrain_dir='${general.logs_dir}model_101/fold_${training.fold}/'

# Continue finetuning with 512x512
python train.py --multirun \
model.model_id=2 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet50 \
training.max_epochs=20 \
scheduler=plateau \
training.pretrain_dir='${general.logs_dir}model_1/fold_${training.fold}/' \
model.input_size=[512,512] \
training.batch_size=24
# ***************************************

# Pretrain on bad labels
python train.py --multirun \
model.model_id=102 \
model.architecture_name=resnet101 \
training.fold=0,1,2,3,4 \
data_mode=bad_quality \
training.max_epochs=10 \
callbacks.model_checkpoint.save_last=true \
scheduler=plateau

# Finetune on good labels
python train.py --multirun \
model.model_id=3 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet101 \
training.max_epochs=50 \
scheduler=plateau \
training.pretrain_dir='${general.logs_dir}model_102/fold_${training.fold}/'
# ***************************************

# Label augmentation ***********************
python train.py --multirun \
model.model_id=103 \
model.architecture_name=resnet50 \
training.fold=0,1,2,3,4 \
data_mode=bad_quality \
training.max_epochs=10 \
callbacks.model_checkpoint.save_last=true \
scheduler=plateau \
training.label_augmentation=0.2

python train.py --multirun \
model.model_id=4 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet50 \
training.max_epochs=50 \
scheduler=plateau \
training.pretrain_dir='${general.logs_dir}model_103/fold_${training.fold}/' \
training.label_augmentation=0.1

python train.py --multirun \
model.model_id=104 \
model.architecture_name=resnet101 \
training.fold=0,1,2,3,4 \
data_mode=bad_quality \
training.max_epochs=10 \
callbacks.model_checkpoint.save_last=true \
scheduler=plateau \
training.label_augmentation=0.2

python train.py --multirun \
model.model_id=5 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnet101 \
training.max_epochs=50 \
scheduler=plateau \
training.pretrain_dir='${general.logs_dir}model_104/fold_${training.fold}/' \
training.label_augmentation=0.1

python train.py --multirun \
model.model_id=105 \
model.architecture_name=resnext50_32x4d \
training.fold=0,1,2,3,4 \
data_mode=bad_quality \
training.max_epochs=10 \
callbacks.model_checkpoint.save_last=true \
scheduler=plateau \
training.label_augmentation=0.2

python train.py --multirun \
model.model_id=6 \
training.fold=0,1,2,3,4 \
model.architecture_name=resnext50_32x4d \
training.max_epochs=50 \
scheduler=plateau \
training.pretrain_dir='${general.logs_dir}model_105/fold_${training.fold}/' \
training.label_augmentation=0.1
# ***************************************
