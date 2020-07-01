# Food recognition

## Documentation <a name="documentation"></a>

**Food recognition** is a end-to-end Data Science project that aims to classify types of dishes with their photographs, using Deep Convolutional Neural Network. <br>
Data used here is directly based on the Food-101 dataset (https://www.tensorflow.org/datasets/catalog/food101). <br>
The dataset is balanced but has 101 different classes, which is a quite high number. <br>
In addition, the training dataset is noised and some wrong label are present. <br>
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
- [```install.sh```](install.sh) create and install Anaconda environment.

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
  AccountName: <account_name>, normally generated within your Azure ML workspace
  AccountKey: <account_key>, normally generated within your Azure ML workspace
  EndpointSuffix: "core.windows.net"
CONTAINER_NAME: Main data container
WORKSPACE_NAME: Workspace name for Azure ML
DATASTORE_NAME: Datastore name for Azure ML
IMAGE_NAME: Docker base image name
COMPUTE_NAME: Training cluster name 
COMPUTE_NAME_INFERENCE: Inference cluster name
```

## How to use <a name="how_to_use"></a>

### Azure set up

Prerequisites: you must have a valid Azure subscription with sufficient rights.

1\) Create a resource group

<img src="/images/resource_group.png"  width="100%" height="100%"> <br>

2\) Create a Machine Learning workspace

<img src="/images/azure_ml_1.png"  width="100%" height="100%"> <br>
<img src="/images/azure_ml_2.png"  width="100%" height="100%"> <br>
<img src="/images/azure_ml_3.png"  width="100%" height="100%"> <br>

At this point, a Storage Account has been created during the process, and you should be able to fill the ACCOUNT_NAME and ACCOUNT_KEY fields in [```config/development.yml```](config/development.yml).

Training cluster creation:

<img src="/images/azure_ml_4.png"  width="100%" height="100%"> <br>

Inference AKS cluster creation:

<img src="/images/azure_ml_5.png"  width="100%" height="100%"> <br>

### Get the data

The following script will get and extract the dataset from a Google Drive endpoint to your Azure Data Storage.

```shell
$ python -m src.download_and_store_data
```

A container named according to the value you set for the field CONTAINER_NAME in [```config/development.yml```](config/development.yml) will be created, and you should be able to acknowledge the transferred data on the Azure portal:

<img src="/images/storage.png"  width="100%" height="100%"> <br>

### Launch a training experiment

```shell
$ python -m src.experiment
```

Endpoint to launch an experiment on AzureML. <br>
Required configuration read from [```src/config.py```](src/config.py) file, 
as well as the entry training script (default: [```src/classification/training.py```](src/classification/training.py)).

#### Regular experiment

If the configuration parameter ```hyperdrive``` is set to False, a regular training experiment is launched, with the hyperparameters fixed in [```src/config.py```](src/config.py).
The underlying TensorFlow model will be saved inside the experiment run scope.

<img src="/images/experiment.png"  width="100%" height="100%"> <br>

#### HyperDrive experiment

If the configuration parameter ```hyperdrive``` is set to True, an HyperDrive hyperparameter search will be performed. <br>
Several major hyperparameters can thus be tuned automatically, here is a non-exhaustive list:
- the image resolution given to the model
- the number of freezed convolutional layers
- the number of additional dense layers
- ...

Here is an exemple with 15 differently parametrized jobs:

<img src="/images/hyperdrive.png"  width="100%" height="100%"> <br>
<img src="/images/hyperdrive_metrics.png"  width="100%" height="100%"> <br>

We can then choose our best run so far according to the primary metric we wanted to maximize:
<img src="/images/metrics.png"  width="100%" height="100%"> <br>

We manage to have a final test accuracy of **81.75%**, and **84.37%** using Test Time Augmentation (TTA). <br>
TTA means we apply several different transformations to a single image, and average the predictions of the model for all of them. That give us a more reliable prediction at the expense of a higher inference time.

### Deploy a trained model on Azure Kubernetes Service

Once we are satisfied with a given terminated experiment, we can register the underlying model and deploy it through AKS. <br>
To achieve this, we need to know the ```run_id```:

<img src="/images/hyperdrive_run.png"  width="100%" height="100%"> <br>

```shell
$ python -m app.register_and_deploy --run-id HD_0224446b-525f-41e9-8852-59d6a6e4f3c3_8
```