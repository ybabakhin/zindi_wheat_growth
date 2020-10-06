# 1st Place Solution for the Zindi CGIAR Wheat Growth Stage Challenge
[Competition website](https://zindi.africa/competitions/cgiar-wheat-growth-stage-challenge)

The problem is to estimate the growth stage of a wheat crop based on an image sent in by the farmer. Model must take in an image and output a prediction for the growth stage of the wheat shown, on a scale from 1 (crop just showing) to 7 (mature crop).

## Instructions to run the code

### System Requirements
The following system requirements should be satisfied:
* OS: Ubuntu 16.04
* Python: 3.6
* CUDA: 10.1
* cudnn: 7
* [pipenv](https://github.com/pypa/pipenv) (`pip install pipenv`)

The training has been done on 2 x GeForce RTX 2080 (training time of the final ensemble is about 15 hours). The batch sizes are selected accordingly.
* Change the list of available GPUs in `./conf/config.yaml`. The parameter is called `gpu_list`
* Override the batch size if needed in `train.sh` and `inference.sh` providing `training.batch_size` argument 

### Environment Setup
1. Navigate to the project directory
2. Run `pipenv install --system --deploy --ignore-pipfile` to install dependencies
3. Run `pipenv shell` to activate the environment

### Data Setup
1. Download data from the competition website and save it to the `./data/` directory.
2. Unzip `Images.zip` there: `unzip Images.zip`

### Best ensemble submission
* The best Private LB submission (0.39949 RMSE) is available in `./lightning_logs/best_model.csv`

### Best ensemble inference
* Download [zindi_wheat_weights.zip](https://drive.google.com/file/d/1gzhfQGSzi4GFMPMsB38P1m3wxWLV2UZw/view?usp=sharing) to the project directory
* Unzip them there: `unzip zindi_wheat_weights.zip`
* For inference, run `./inference.sh`. Final ensemble predictions will be saved to `./lightning_logs/2_3_4_5_6_ens.csv`

### Train the model from scratch
* To start training the model from scratch, run `./train.sh` (takes about 15 hours on 2x2080)
* Afterwards, `./inference.sh` could be run

## Solution Description
The dataset has two sets of labels: `bad` and `good` quality, but test dataset consists only of good quality labels.
First of all, there is no clear correspondance between bad and good labels (good labels contain only five classes: `2, 3, 4, 5, 7`).
Secondly, bad and good quality images could be easily distinguished using a simple binary classifier. So, they come from the different distributions. Looking at the Grad-CAM of such a model suggests that the major difference between two sets of images is these white sticks (poles):
![](imgs/gradcam_quality.png?raw=true "Grad-CAM")

That's why training process consists of 2 steps:
1. Pre-train the model on the mix of bad and good quality labels
2. Finetune the model only on the good quality labels

### Single best model
Best single model (average of 5 folds on the test set) achieves 0.40327 RMSE on Private LB.

#### Model hyperparameters
* Architecture: ResNet101
* Problem type: classification
* Loss: cross entropy
* FC Dropout probability: 0.3
* Input size: (256, 256)
* Predicted probabilities are multiplied by the class labels and summed up

#### Augmentations
The median image size in the data is about `180 x 512`. For preprocessing, firstly image is padded to `img_width // 2 x img_width` and then is resized to `256 x 256`. Augmentations list includes:
* Horizontal flips
* RandomBrightnessContrast
* ShiftScaleRotate
* Imitating additional white sticks (poles) on the images
* Label augmentation (changing class labels to the neighbor classes with a low probability)
* Use horizontal flips as TTA

A couple of examples:
![](imgs/augmentation_1.png?raw=true "Augmentation 1")
![](imgs/augmentation_2.png?raw=true "Augmentation 2")

#### Training process
1. Pre-train on mix of good and bad labels for 10 epochs
2. Fine-tune on good labels for 50 epochs with reducing learning rate on plateau

### Ensemble
Ensembling multiple models worked pretty well in this problem. My final solution is just an average of five 5-fold models (25 checkpoints overall):
* Architectures: ResNet50, ResNet101, ResNeXt50
* Input sizes: `256x256` and `512x512`
* It achieved 0.39949 RMSE on Private LB

### What didn't work
* Various options of pseudolabels. Generating pseudolabels for bad quality label and test set; pre-training; mixing with actual labels, etc.
* MixUp and CutMix augmentations
* EffecientNet architectures
* Treating the problem as a regression (didn't help even in the ensemble)
* Stacking over the first level models predictions
