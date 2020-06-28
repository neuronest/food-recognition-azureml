"""
Endpoint to download source data from Google Drive, extract it, and load it to Azure Blob Storage.
"""
import yaml
import os
import zipfile
import shutil
from os.path import join, dirname
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from src.utils import (
    download_file_from_google_drive,
    write_image_azure,
    read_image_local,
)
from src.config import GeneralConfig, PathsConfig

with open(join(dirname(__file__), join("..", "config", "development.yml"))) as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    temporary_file_location = "/tmp/data"
    container_name = settings["CONTAINER_NAME"]
    dataset_name = settings["DATASET_NAME"]
    connexion_string = ";".join(
        map(
            lambda x: "=".join([x, settings["CREDENTIALS"][x]]), settings["CREDENTIALS"]
        )
    )
    blob_service_client = BlobServiceClient.from_connection_string(connexion_string)
    print("Downloading source data...")
    download_file_from_google_drive(
        settings["DATASET_DRIVE_ID"], temporary_file_location + ".zip"
    )
    print("Source data downloaded!")
    print("Extracting source data...")
    with zipfile.ZipFile(temporary_file_location + ".zip", "r") as zip_ref:
        zip_ref.extractall(temporary_file_location)
    os.remove(temporary_file_location + ".zip")
    print("Source data extracted!")
    print("Uploading data to Azure Blob Storage...")
    try:
        blob_service_client.create_container(container_name)
    except ResourceExistsError:
        pass
    classes = "\n".join(
        sorted(
            os.listdir(
                os.path.join(
                    temporary_file_location,
                    settings["DATASET_NAME"],
                    PathsConfig.data_train,
                )
            )
        )
    )
    blob_object = blob_service_client.get_blob_client(
        container_name, PathsConfig.classes
    )
    blob_object.upload_blob(classes)
    for root, dirs, files in os.walk(temporary_file_location):
        for file in files:
            blob_name = os.path.relpath(
                os.path.join(root, file),
                os.path.join(temporary_file_location, dataset_name),
            )
            filename = os.path.join(root, file)
            if filename.endswith(GeneralConfig.image_extension):
                image = read_image_local(filename)
                write_image_azure(
                    blob_service_client,
                    container_name,
                    blob_name,
                    image,
                    overwrite=False,
                )
    shutil.rmtree(temporary_file_location)
    print("Data uploaded!")
