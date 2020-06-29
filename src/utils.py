import os
import json
import requests
import numpy as np
import tensorflow as tf
from typing import Union, Optional
from azure.storage.blob import BlobServiceClient
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.preprocessing.image import load_img, save_img

try:
    from config import GeneralConfig, PathsConfig
except ImportError:
    from .config import GeneralConfig, PathsConfig


def remove_extension_from_filename(filename: str) -> str:
    """
    Return a filename without its extension
    """
    return os.path.splitext(filename)[0]


def get_extension_from_filename(filename: str) -> str:
    """
    Return the extension from a given filename
    """
    return os.path.splitext(filename)[-1]


def read_image_local(path: str, target_size: bool = None) -> np.ndarray:
    """
    Image reader from local.
    The image is resized to target_size if needed
    """
    if target_size is not None:
        return np.asarray(load_img(path, target_size=target_size))
    else:
        return np.asarray(load_img(path))


def write_image_local(path: str, image: np.ndarray) -> None:
    """
    Image writer to local.
    """
    save_img(path, image)


def read_text_azure(
    blob_service_client: BlobServiceClient, container: str, blob: str,
) -> str:
    return (
        blob_service_client.get_blob_client(container, blob)
        .download_blob()
        .readall()
        .decode("utf-8")
    )


def write_text_azure(
    blob_service_client: BlobServiceClient,
    container: str,
    blob: str,
    text: str,
    overwrite=False,
) -> None:
    blob_object = blob_service_client.get_blob_client(container, blob)
    blob_object.upload_blob(text, overwrite=overwrite)


def read_image_azure(
    blob_service_client: BlobServiceClient,
    container: str,
    blob: str,
    target_size: tuple = None,
    to_numpy: bool = False,
    to_float32: bool = False,
    channels: int = 3,
) -> Union[np.ndarray, tf.python.framework.ops.EagerTensor]:
    """
    Image reader from Azure Blob Storage.
    The image is resized to target_size if resize_image is True, and can be casted to numpy array if needed
    """
    image_bytes = (
        blob_service_client.get_blob_client(container, blob).download_blob().readall()
    )
    image = tf.image.decode_image(image_bytes, channels=channels)
    if target_size is not None:
        image = tf.image.resize(image, size=target_size)
    if to_float32:
        image = tf.cast(image, dtype="float32")
    if to_numpy:
        image = image.numpy()
    return image


def write_image_azure(
    blob_service_client: BlobServiceClient,
    container: str,
    blob: str,
    image: Union[np.ndarray, tf.python.framework.ops.EagerTensor],
    target_size: tuple = None,
    from_float32: bool = False,
    overwrite=False,
) -> None:
    """
    Image writer on Azure Blob Storage
    """
    blob_object = blob_service_client.get_blob_client(container, blob)
    extension = get_extension_from_filename(blob)
    if target_size is not None:
        image = tf.image.resize(image, size=target_size)
    if from_float32:
        image = tf.cast(image, dtype="uint8")
    if extension.lower() == ".jpg":
        encoded_image = tf.image.encode_jpeg(image).numpy()
    elif extension.lower() == ".png":
        encoded_image = tf.image.encode_png(image).numpy()
    else:
        raise NotImplementedError
    blob_object.upload_blob(encoded_image, overwrite=overwrite)


def service_query(
    service_uri: str,
    service_key: str,
    image: np.ndarray,
    container: str,
    blob: str,
    number_of_passes: int = 1,
    return_probabilities: bool = False,
):
    encoded_image = list(tf.image.encode_jpeg(image))
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(service_key),
    }
    data = {
        "image": encoded_image,
        "container": container,
        "blob": blob,
        "number_of_passes": number_of_passes,
        "return_probabilities": return_probabilities,
    }
    data_raw = bytes(json.dumps({"data": data}), encoding="utf8")
    response = requests.post(service_uri, data_raw, headers=header)
    return response.content


def download_file_from_google_drive(file_id: str, destination_path: str) -> None:
    # inspired from https://stackoverflow.com/a/39225039
    def get_confirm_token(response: requests.models.Response) -> Optional[str]:
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(
        response: requests.models.Response, destination: str, chunk_size: int = 32768
    ) -> None:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:  # filter out keep-alive new chunksqs
                    f.write(chunk)

    google_api_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(google_api_url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(
            google_api_url, params={"id": file_id, "confirm": token}, stream=True
        )
    save_response_content(response, destination_path)
