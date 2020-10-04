#  Place Solution for the Zindi CGIAR Wheat Growth Stage Challenge
[Competition website](https://zindi.africa/competitions/cgiar-wheat-growth-stage-challenge)

The problem is to estimate the growth stage of a wheat crop based on an image sent in by the farmer. Model must take in an image and output a prediction for the growth stage of the wheat shown, on a scale from 1 (crop just showing) to 7 (mature crop).

## Instructions to run the code

### Environment

### Data Setup
1. Download data from the competition page and save it to the `./data/` directory.
2. Unzip `Image.zip`

### Weights Setup

### Docker Setup

### Train the model
* Change the list of available GPUs in `./conf/config.yaml`. The parameter is called `gpu_list`.
* To start training the model simply run `./run.sh` script

## Solution Description
The dataset has two sets of labels: `bad` and `good` quality, but test dataset consists only of good quality labels.
First of all, there is no clear correspondance between bad and good labels (good labels contain only five classes: `2, 3, 4, 5, 7`).
Secondly, bad and good quality images could be easily distinguished using a simple binary classifier. So, they come from the different distributions. Looking at the Grad-CAM of such a model suggests that the major difference between two sets of images is these white sticks (poles):
![](imgs/gradcam_quality.png?raw=true "Grad-CAM")

That's why my training process consists of 2 steps:
1. Pre-train the model on the mix of bad and good quality labels
2. Finetune the model only on the good quality labels

### Single best model
Best single model (average of 5 folds on the test set) achieves 0.42679 RMSE on local validation and 0.42818 on Public LB.

#### Model hyperparameters
* Architecture: ResNet50
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
![](imgs/augmentation_1.png?raw=true "augmentation_1")
![](imgs/augmentation_2.png?raw=true "augmentation_2")

#### Training process
1. Pre-train on mix of good and bad labels for 10 epochs
2. Fine-tune on good labels for 50 epochs with reducing learning rate on plateau
3. Fine-tune for 10 epochs more with cosine annealing and a smaller learning rate

### Ensemble
Ensembling multiple models worked pretty well in this problem. My final solution is just an average of twelve 5-fold models (60 checkpoints overall):
* Architectures: ResNet50, ResNet101, ResNeXt50, DenseNet169
* Input sizes: `256x256` and `512x512`
* It achieved 0.406 RMSE on local validation and 0.42135 on Public LB

### What didn't work
* Various options of pseudolabels. Generating pseudolabels for bad quality label and test set; pre-training; mixing with actual labels, etc.
* MixUp and CutMix augmentations
* EffecientNet architectures
* Treating the problem as a regression (didn't help even in the ensemble)
* Stacking of 1st level models predictions
