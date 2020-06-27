
# Dessert recognition


## Documentation <a name="documentation"></a>

**Dessert recognition** is a Data Science project that aims to classify types of dessert using their photographs.

## Repository Structure <a name="repository-structure"></a>

```
|_ src/
|____ classification/
|__________________ architecture/
|______________________________ ...
|__________________ deployment/
|____________________________ ...
|__________________ training.py
|__________________ inference.py
|_____config.py
|_____download_and_store_data.py
|____ experiment.py
|____ plot.py
|____ register_and_deploy.py
|____ utils.py
|_ config/
|_______ ...
|_ ...
```

- [```src/```](src) is the repertory with all project code
- [```config/```](config) is the repertory with config files, especially related to Azure paths


## Installation <a name="installation"></a>


```shell**
git clone https://github.com/neuronest/dessert_recognition.git
cd dessert_recognition
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
ACCOUNT_NAME: <account_name>  
ACCOUNT_KEY: <account_key>  
CONTAINER_NAME: Main data container
WORKSPACE_NAME: Workspace name for Azure ML
DATASTORE_NAME: Datastore name for Azure ML
COMPUTE_NAME: Machine Learning Compute name 
COMPUTE_NAME_INFERENCE: Kubernetes Service name
```
