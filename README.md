
# Food recognition


## Documentation <a name="documentation"></a>

**Food recognition** is a end-to-end Data Science project that aims to classify types of dishes with their photographs, using Deep Convolutional Neural Network. <br>
Data used here is directly based on the Food-101 dataset (https://www.tensorflow.org/datasets/catalog/food101). <br>
Of course, a food classification task is an excuse: the same processes can be applied on a wide variety of computer vision tasks. <br>
This repository contains the entire code to:
- Collect the data
- Train a ResNet model using TensorFlow 2.x on Azure Machine Learning
- Deploy and expose a trained model on Azure Kubernetes Service

## Repository Structure <a name="repository-structure"></a>

```
|_ config/
|_______ ...
|_ src/
|____ classification/
|__________________ architecture/
|______________________________ ...
|__________________ deployment/
|____________________________ ...
|__________________ training.py
|_____config.py
|_____download_and_store_data.py
|____ experiment.py
|____ plot.py
|____ register_and_deploy.py
|____ utils.py
|_ .flake8
|_ .gitignore
|_ .pre-commit-config.yaml
|_ enable_env.sh
|_ format.sh
|_ install.sh
|_ LICENSE
|_ README.md
|_ requirements.txt
```

- [```config/```](config) repertory with config files, especially related to Azure paths et credentials.
- [```src/```](src) repertory with all project code.
- [```src/classification/```](src/classification) repertory containing implementations for the classification problem.
- [```src/classification/architectures```](src/classification/architectures) repertory containing TensorFlow code for the ResNet50 model.
- [```src/classification/deployment```](src/classification/deplomyent) repertory related to model serving on Azure Kubernetes Service.
- [```src/classification/training.py```](src/classification/training.py) entry script classification model training, should not be used directly but within an experiment.
- [```src/config.py```](src/config.py) main configuration script for this project.
- [```src/download_and_store_data.py```](src/download_and_store_data.py) script reaching data from Google Drive and loading it to Azure Blob Storage.
- [```src/experiment.py```](src/experiment.py) main script to launch an Azure ML experiment.
- [```src/plot.py```](src/plot.py) contains plotting functions.
- [```src/register_and_deploy.py```](src/register_and_deploy.py) main script to register a trained model and expose it using AKS.
- [```src/utils.py```](src/utils.py) contains utility functions, especially readers and writers.
- [```enable_env.sh```](enable_env.sh) enable Anaconda environment.
- [```install.sh```](enable_env.sh) create and install Anaconda environment.

## Installation <a name="installation"></a>


```shell**
git clone https://github.com/neuronest/food_recognition.git
cd food_recognition
source install.sh
```

## Configuration <a name="config"></a>

### Credentials and resources names filling

Be sure the [```config/development.yml```](config/development.yml) file is correctly filled:

```
DATASET_NAME: Dataset name
DATASET_DRIVE_ID: <google_drive_file_id>  
SUBSCRIPTION_ID: <subscription_id>  
RESOURCE_GROUP: <resource_group>  
CREDENTIALS:
  DefaultEndpointsProtocol: "https"
  AccountName: Azure Storage account name
  AccountKey: Azure Storage account key
  EndpointSuffix: "core.windows.net"
CONTAINER_NAME: Main data container
WORKSPACE_NAME: Workspace name for Azure ML
DATASTORE_NAME: Datastore name for Azure ML
IMAGE_NAME: Docker base image name
COMPUTE_NAME: Training cluster name 
COMPUTE_NAME_INFERENCE: Inference cluster name
```
