"""
Endpoint script for classification model deployment
Should not be used directly but as entry script in src/register_and_deploy.py
"""

import os
import sys
import json
import yaml
import pickle
import numpy as np
import tensorflow as tf
from os.path import join, dirname, abspath
from tensorflow.keras.models import load_model
from azure.storage.blob import BlobServiceClient

root = dirname(dirname(dirname(dirname(abspath(__file__)))))
directories = [
    dirname(dirname(dirname(abspath(__file__)))),
    dirname(dirname(abspath(__file__))),
    dirname(abspath(__file__)),
]
for dir_path in directories:
    sys.path.append(dir_path)

from config import ModelConfig, PathsConfig  # noqa: E402
from utils import read_image_azure, read_text_azure  # noqa: E402

with open(join(root, "config", "development.yml")) as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

connexion_string = ";".join(
    map(lambda x: "=".join([x, settings["CREDENTIALS"][x]]), settings["CREDENTIALS"])
)
blob_service_client = BlobServiceClient.from_connection_string(connexion_string)
input_dim = ModelConfig.pretrained_resnet50_hyperparams["input_dim"]


def init():
    global augmented_generator_schema
    global generator_schema
    global model
    global classes
    with open(
        join(
            os.getenv("AZUREML_MODEL_DIR"), PathsConfig.augmented_image_generator_path
        ),
        "rb",
    ) as fp:
        augmented_generator_schema = pickle.load(fp)
    with open(
        join(os.getenv("AZUREML_MODEL_DIR"), PathsConfig.image_generator_path), "rb"
    ) as fp:
        generator_schema = pickle.load(fp)
    model_path = join(os.getenv("AZUREML_MODEL_DIR"), PathsConfig.model_directory)
    model = load_model(model_path)
    classes = read_text_azure(
        blob_service_client, settings["CONTAINER_NAME"], PathsConfig.classes
    ).split("\n")


def run(raw_data):
    data = json.loads(raw_data)["data"]
    image, container, blob, number_of_passes, return_probabilities = (
        data.get("image", None),
        data.get("container", None),
        data.get("blob", None),
        data.get("number_of_passes", 1),
        data.get("return_probabilities", False),
    )
    if image is not None:
        image = tf.image.decode_image(bytes(image), channels=3)
    elif container is not None and blob is not None:
        image = read_image_azure(blob_service_client, container, blob, to_float32=True)
    else:
        print("No data received!")
        response = {"probabilities": None, "prediction_index": None, "prediction": None}
        return response
    if number_of_passes == 1:
        generator = generator_schema.flow(
            tf.image.resize(tf.expand_dims(image, axis=0), size=(input_dim, input_dim))
        )
        y_pred_proba = model.predict(generator)
    else:
        augmented_generator = augmented_generator_schema.flow(
            tf.image.resize(tf.expand_dims(image, axis=0), size=(input_dim, input_dim))
        )
        y_pred_proba = np.mean(
            model.predict(
                np.concatenate(
                    [augmented_generator.next() for _ in range(number_of_passes)]
                )
            ),
            axis=0,
            keepdims=True,
        )
    prediction_index = int(np.argmax(y_pred_proba, axis=1)[0])
    prediction = classes[prediction_index]
    y_pred_proba_list = y_pred_proba.tolist() if return_probabilities else None
    response = {
        "probabilities": y_pred_proba_list,
        "prediction_index": prediction_index,
        "prediction": prediction,
    }
    return response
