#!/usr/bin/env bash

# Base pretraining (166, 177, 178) ***********************
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
#training.pretrain_dir='${general.logs_dir}model_923/fold_${training.fold}/'
# ***************************************

# Continue training with 512x512 ***********************
#python train.py --multirun \
#model.model_id=174 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=20 \
#scheduler=plateau \
#training.pretrain_dir='${general.logs_dir}model_166/fold_${training.fold}/' \
#training.batch_size=24
# ***************************************

# Improved pretraining (210, 211, 214) ***********************
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
#training.pretrain_dir='${general.logs_dir}model_944/fold_${training.fold}/' \
#training.label_augmentation=0.1
# ***************************************

# Continue training with 512x512 ***********************
#python train.py --multirun \
#model.model_id=215 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=20 \
#scheduler=plateau \
#training.pretrain_dir='${general.logs_dir}model_210/fold_${training.fold}/' \
#training.batch_size=24 \
#model.input_size=[512,512]
# ***************************************

# Improved pretraining 512x512 (220 (bs 24), 221 (bs 16), 222 (bs 20) ***********************
#python train.py --multirun \
#model.model_id=952 \
#model.architecture_name=resnext50_32x4d \
#training.fold=0,1,2,3,4 \
#data_mode=bad_quality \
#training.max_epochs=10 \
#callbacks.model_checkpoint.save_last=true \
#scheduler=plateau \
#training.batch_size=20 \
#training.label_augmentation=0.2 \
#model.input_size=[512,512]
#
#python train.py --multirun \
#model.model_id=222 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnext50_32x4d \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='${general.logs_dir}model_952/fold_${training.fold}/' \
#training.batch_size=20 \
#training.label_augmentation=0.1 \
#model.input_size=[512,512]
# ***************************************

# Continue training 210, 211, 214 ***********************
#python train.py --multirun \
#model.model_id=224 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_210/fold_${training.fold}/' \
#training.lr=1e-6
#
#python train.py --multirun \
#model.model_id=226 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet101 \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_211/fold_${training.fold}/' \
#training.lr=1e-6
#
#python train.py --multirun \
#model.model_id=228 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnext50_32x4d \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_214/fold_${training.fold}/' \
#training.lr=1e-6
# ***************************************
#
# Continue training 220, 221, 222 (232, 233, 234) ***********************
#python train.py --multirun \
#model.model_id=234 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnext50_32x4d \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_222/fold_${training.fold}/' \
#training.lr=1e-6 \
#model.input_size=[512,512] \
#training.batch_size=20
# ***************************************

# Inference ***********************
#python test.py --multirun \
#model.model_id=224 \
#model.architecture_name=resnet50 \
#testing.mode=valid,pseudo,test
#
#python test.py --multirun \
#model.model_id=226 \
#model.architecture_name=resnet101 \
#testing.mode=valid,pseudo,test
#
#python test.py --multirun \
#model.model_id=228 \
#model.architecture_name=resnext50_32x4d \
#testing.mode=valid,pseudo,test
#
#python test.py --multirun \
#model.model_id=210 \
#model.architecture_name=resnet50 \
#testing.mode=valid,test
#
#python test.py --multirun \
#model.model_id=211 \
#model.architecture_name=resnet101 \
#testing.mode=valid,test
#
#python test.py --multirun \
#model.model_id=214 \
#model.architecture_name=resnext50_32x4d \
#testing.mode=valid,test
#
#python test.py --multirun \
#model.model_id=220,232 \
#model.architecture_name=resnet50 \
#testing.mode=valid,test \
#model.input_size=[512,512]
#
#python test.py --multirun \
#model.model_id=221,233 \
#model.architecture_name=resnet101 \
#testing.mode=valid,test \
#model.input_size=[512,512]
#
#python test.py --multirun \
#model.model_id=222,234 \
#model.architecture_name=resnext50_32x4d \
#testing.mode=valid,test \
#model.input_size=[512,512]

#python ensemble.py --multirun \
#ensemble.model_ids=[224,226,228,210,211,214,220,221,222,232,233,234] \
#testing.mode=valid,test
# ***************************************

# Pseudolabels ***********************
#python train.py --multirun \
#model.model_id=242 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='${general.logs_dir}model_224/fold_${training.fold}/' \
#data_mode.pseudolabels_path='${general.logs_dir}224_226_228_pseudo_fold_${training.fold}.csv' \
#training.label_augmentation=0.1
# ***************************************
#
# Crops ***********************
#python train.py --multirun \
#model.model_id=958 \
#model.architecture_name=resnet50 \
#training.fold=0,1,2,3,4 \
#data_mode=bad_quality \
#training.max_epochs=10 \
#callbacks.model_checkpoint.save_last=true \
#scheduler=plateau \
#training.label_augmentation=0.2 \
#model.crop_method=crop

#python train.py --multirun \
#model.model_id=243 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='${general.logs_dir}model_958/fold_${training.fold}/' \
#training.label_augmentation=0.1 \
#model.crop_method=crop

#python train.py --multirun \
#model.model_id=249 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_243/fold_${training.fold}/' \
#training.lr=1e-6 \
#model.crop_method=crop
# ***************************************

# Regression ***********************
#python train.py --multirun \
#model.model_id=244 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='${general.logs_dir}model_943/fold_${training.fold}/' \
#model.regression=true \
#data_mode.num_classes=1
# ***************************************

# CutMix + Mixup ***********************
#python train.py --multirun \
#model.model_id=960 \
#model.architecture_name=resnet50 \
#training.fold=0,1,2,3,4 \
#data_mode=bad_quality \
#training.max_epochs=10 \
#callbacks.model_checkpoint.save_last=true \
#scheduler=plateau \
#training.label_augmentation=0.1 \
#training.augmentations=hard \
#model.dropout=0.4 \
#training.mixup=0.2
#
#python train.py --multirun \
#model.model_id=247 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=50 \
#scheduler=plateau \
#training.pretrain_dir='${general.logs_dir}model_960/fold_${training.fold}/' \
#training.augmentations=hard \
#model.dropout=0.4 \
#training.mixup=0.2
#
#python train.py --multirun \
#model.model_id=248 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_247/fold_${training.fold}/' \
#training.lr=1e-6
# ***************************************

# Save 2 checkpoints for finetuning ***********************
#python train.py --multirun \
#model.model_id=256 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet50 \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_210/fold_${training.fold}/' \
#training.lr=1e-6 \
#callbacks.model_checkpoint.save_top_k=2

#python train.py --multirun \
#model.model_id=257 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnet101 \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_211/fold_${training.fold}/' \
#training.lr=1e-6 \
#callbacks.model_checkpoint.save_top_k=2
#
#python train.py --multirun \
#model.model_id=258 \
#training.fold=0,1,2,3,4 \
#model.architecture_name=resnext50_32x4d \
#training.max_epochs=10 \
#training.pretrain_dir='${general.logs_dir}model_214/fold_${training.fold}/' \
#training.lr=1e-6 \
#callbacks.model_checkpoint.save_top_k=2
# ***************************************

python test.py --multirun \
model.model_id=254 \
model.architecture_name=densenet169 \
testing.mode=valid,pseudo,test
