"""
Endpoint to launch an experiment on AzureML.
"""

import yaml
from os.path import join, dirname
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from azureml.train.estimator import Estimator
from azureml.core import Workspace, Datastore, Experiment
from azureml.train.hyperdrive import (
    RandomParameterSampling,
    BanditPolicy,
    HyperDriveConfig,
    PrimaryMetricGoal,
    choice,
)

from src.config import GeneralConfig, PathsConfig, HyperdriveConfig

with open(join(dirname(__file__), join("..", "config", "development.yml"))) as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    container_name = settings["CONTAINER_NAME"]
    datastore_name = settings["DATASTORE_NAME"]
    connexion_string = ";".join(
        map(
            lambda x: "=".join([x, settings["CREDENTIALS"][x]]), settings["CREDENTIALS"]
        )
    )
    blob_service_client = BlobServiceClient.from_connection_string(connexion_string)

    ws = Workspace(
        settings["SUBSCRIPTION_ID"],
        settings["RESOURCE_GROUP"],
        settings["WORKSPACE_NAME"],
    )

    try:
        blob_service_client.get_container_client(
            container_name
        ).get_container_properties()
    except ResourceNotFoundError:
        raise ResourceNotFoundError(
            "Data is not properly loaded into Azure Blob Storage, please run download_and_store_data.py first"
        )

    ds = Datastore.register_azure_blob_container(
        workspace=ws,
        datastore_name=datastore_name,
        container_name=container_name,
        account_name=blob_service_client.account_name,
        account_key=settings["CREDENTIALS"]["AccountKey"],
    )
    if settings["COMPUTE_NAME"] in ws.compute_targets:
        compute_target = ws.compute_targets[settings["COMPUTE_NAME"]]
    else:
        compute_target = None
    assert compute_target is not None, "No compute target has been found"

    experiment_name = "-".join(
        [settings["DATASTORE_NAME"], GeneralConfig.architecture_type]
    )
    if GeneralConfig.hyperdrive:
        experiment_name += "-" + "hyperdrive"
    exp = Experiment(workspace=ws, name=experiment_name)
    source_directory = dirname(__file__)

    est = Estimator(
        source_directory=source_directory,
        script_params={"--data-folder": ds.as_mount()},
        compute_target=compute_target,
        pip_packages=GeneralConfig.pip_packages,
        entry_script=PathsConfig.entry_script,
        use_gpu=True,
        custom_docker_image=settings["IMAGE_NAME"],
    )
    if GeneralConfig.hyperdrive:
        if GeneralConfig.architecture_type == "PretrainedResNet50":
            hyperparams_space = HyperdriveConfig.pretrained_resnet50_hyperparams_space
        else:
            raise NotImplementedError
        hyperparams_space_format = {
            parameter: choice(parameter_range)
            for parameter, parameter_range in hyperparams_space.items()
        }
        parameters_sampling = RandomParameterSampling(hyperparams_space_format)
        policy = BanditPolicy(
            evaluation_interval=HyperdriveConfig.evaluation_interval,
            slack_factor=HyperdriveConfig.slack_factor,
        )
        hdc = HyperDriveConfig(
            estimator=est,
            hyperparameter_sampling=parameters_sampling,
            policy=policy,
            primary_metric_name="Accuracy",
            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
            max_total_runs=HyperdriveConfig.max_total_runs,
            max_concurrent_runs=HyperdriveConfig.max_concurrent_runs,
        )
        run = exp.submit(hdc)
    else:
        run = exp.submit(est)
