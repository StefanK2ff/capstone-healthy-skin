# Team DermaNerds Capstone Project
![Banner goes here](./images/DermaNerds_nf.png)


## Convolutional Neural Network Image Classification of Pigmented Skin Lesions 
![GitHub contributors](https://img.shields.io/github/contributors/StefanK2ff/capstone-healthy-skin)
![Static Badge](https://img.shields.io/badge/Lifescience-yellow)
![Static Badge](https://img.shields.io/badge/medicine-green)
![Static Badge](https://img.shields.io/badge/skin_cancer-violet)
![Static Badge](https://img.shields.io/badge/ResNet50-blue)

This is the capstone project for the Datasciene bootcamp of neuefische GmbH ffm-ds-23, where the task was to utilize the knowledge from the bootcamp to solve an everyday problem. 

The teams were self organized and the topics were chosen by the trainees themselfes. Team DermaNerds formed from the passion of using neural networks in Lifescience and Healthcare applications and chose the HAM10000 ("Human Against Machine with 10000 training images") dataset for this project as a stellar example of image classification in medicine. [SOURCE](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

The HAM10000 dataset addresses the challenge of training neural networks for automated skin lesion diagnosis by providing a diverse collection of 10,015 dermatoscopic images from different populations, modalities, and diagnostic categories, making it a valuable resource for academic machine learning research in this field.

ResNet-50 is a deep convolutional neural network known for its superior accuracy, reduced overfitting, and ease of transfer learning. After comparing to other CNN architectures like VGG16, MobileNet and others ResNet50 showed the best performence on the dataset. All final evaluations were performed on three classes namely "benign", "malignant", "non-neoplastic and the classes were grouped via the metadata_engineering notebook.

### Workflow
#### Environment Setup

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Image Folder path
- unpack all jpgs from HAM10000_images_part1.zip (5000 JPEG files) &
HAM10000_images_part2.zip (5015 JPEG files) to:  
./Capstone-Healthy-Skin/data/jpg

#### Metadata Folder path
- unpack HAM10000_metadata.tab to:
./Capstone-Healthy-Skin/data
- The notebook "metadata_engineering" can be used to group lesion type dx into three classes, otherwise 7 classes is the default.

#### Notebook usage

##### 0_image_loader_Albumentation
* This notebook utilizes the [Albumentation](https://albumentations.ai/) library to resample the classes with augmented images. Possible settings for Target_label are: dx(7 classes), dx_binary(2 classes), dx_tertiary(3 classes)
* The image_loader will automatically reduce/increase samples per class as stated in MAX_SAMPLES_TRAIN
  
##### 1_resnet50_final_setup
* This notebook will preprocces the train data e.g. centering and cropping
* It will also randomly augment the train data to improve learning
* It contains the optimized resnet50 model architecture and the model training
* After training the model is evaluated using the modelhelper helperfunction
* Finally the model is saved as an .h5 for future comparisons
  
##### 2_loading_models
* This notebook can load previously saved models for re-evaluation
* It will load all saved models in a given directory for easy comparison

#### Performance
In this 4 week project the following was achieved:

![Evaluation-Metrics go here](./images/results.png)

#### The Team
* [Arathi Ramesh](https://github.com/eigenaarti2)
* [Björn Zimmermann](https://github.com/bjzim)
* [Daniel Marques Rodrigues](https://github.com/Da-MaRo)
* [Janice Pelzer](https://github.com/janicepelzer)
* [Stefan Berkenhoff](https://github.com/StefanK2ff)

#### The Coaches
* [Aljoscha	Wilhelm](https://github.com/JodaFlame)
* [Arjun	Haridas](https://github.com/Ajax121)
* [Evgeny	Savin](https://github.com/EvgenySavin1985)
