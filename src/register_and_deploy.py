"""
Script for deploying a trained model on a AKS cluster.
"""

import sys
import yaml
import argparse
import json
import time
from os.path import join, dirname
from azureml.core import Workspace, Experiment, Run
from azureml.exceptions import ServiceException
from azureml.core.runconfig import CondaDependencies
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AksWebservice

from src.config import GeneralConfig, PathsConfig
from src.classification.deployment.deployment_config import (
    deployment_config,
    conda_packages,
    pip_packages,
)

with open(join(dirname(__file__), join("..", "config", "development.yml"))) as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

subscription_id = settings["SUBSCRIPTION_ID"]
resource_group = settings["RESOURCE_GROUP"]
workspace_name = settings["WORKSPACE_NAME"]
datastore_name = settings["DATASTORE_NAME"]
compute_name = settings["COMPUTE_NAME_INFERENCE"]

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=str, dest="run_id", help="Run id")
parser.add_argument(
    "--test-only", action="store_true", help="Test an already deployed model only",
)
parser.add_argument(
    "--container", type=str, help="Inference container source", default="source",
)
parser.add_argument(
    "--blob",
    type=str,
    help="Inference blob source",
    default="test/apple_pie/101251.jpg",
)
parser.add_argument(
    "--service-name-suffix", type=str, default="", help="model service suffix"
)
args = parser.parse_args()
run_id = args.run_id
test_only = args.test_only
container = args.container
blob = args.blob
service_name_suffix = args.service_name_suffix

experiment_name = "-".join(
    [settings["DATASTORE_NAME"], GeneralConfig.architecture_type]
)
model_name = experiment_name[
    :32
]  # provided model name to Azure must have at most 32 characters
service_name = experiment_name.lower()
service_name = (
    service_name + "-" + service_name_suffix if service_name_suffix else service_name
)
service_name = service_name[
    :32
]  # provided service name to Azure must have at most 32 characters


def test_service(service, container, blob, write_logs=True):
    if write_logs:
        logs = service.get_logs()
        with open("logs.txt", "w") as fp:
            fp.write(logs)
    data = {"container": container, "blob": blob}
    data_raw = bytes(json.dumps({"data": data}), encoding="utf8")
    print("Testing service: {0}".format(service.name))
    print("Container: {0}, blob: {1}".format(container, blob))
    ping = time.time()
    response = service.run(input_data=data_raw)
    print("Elapsed time: {0:.5f}".format(time.time() - ping))
    print("Response: {0}".format(response))


if __name__ == "__main__":
    ws = Workspace(subscription_id, resource_group, workspace_name)
    if test_only:
        service = ws.webservices[service_name]
        test_service(service, container, blob)
        sys.exit()
    exp = Experiment(workspace=ws, name=experiment_name)
    try:
        run = Run(exp, run_id)
    except ServiceException:
        print("Run id not found, exiting...")
        sys.exit()
    model = run.register_model(
        model_name=model_name,
        model_path=PathsConfig.outputs_directory,
        tags={"run_id": run_id},
    )
    cd = CondaDependencies.create()
    for conda_package in conda_packages:
        cd.add_conda_package(conda_package)
    for pip_package in pip_packages:
        cd.add_pip_package(pip_package)
    cd.add_tensorflow_pip_package(core_type="gpu", version="2.2.0")
    cd.save_to_file(
        base_directory="src/classification/deployment", conda_file_path="env.yml",
    )
    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
    else:
        compute_target = None
    assert compute_target is not None
    aks_config = AksWebservice.deploy_configuration(
        **deployment_config, tags={"run_id": run_id}
    )
    inference_config = InferenceConfig(
        runtime="python",
        entry_script="src/classification/deployment/serving.py",
        conda_file="src/classification/deployment/env.yml",
        source_directory=".",
        enable_gpu=True,
        description="food recognition model",
    )
    aks_service = Model.deploy(
        workspace=ws,
        models=[model],
        inference_config=inference_config,
        deployment_config=aks_config,
        deployment_target=compute_target,
        name=service_name,
        overwrite=True,
    )
    aks_service.wait_for_deployment(show_output=True)
