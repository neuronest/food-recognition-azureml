"""
Library containing metrics and inference functions for the classification model
"""

import os
import numpy as np
import pandas as pd
from skimage.io import imread
from os.path import join


def inference(model, chunks, return_probabilities=True):
    """
    Compute counts per class based on a trained classification model
    :param model: Keras model: already trained classification model
    :param chunks: Numpy array: stacked rosebud chunks
    :param return_probabilities: boolean, whether or not to return model probabilities
    :return: (Numpy array: counts per class, Optional Numpy array: model probabilities matrix)
    """
    if normalization:
        chunks /= 255.0
    y_pred_proba = model.predict(chunks)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_no_garbage = y_pred[y_pred != 0]
    pred_counts = dict(zip(*np.unique(y_pred_no_garbage, return_counts=True)))
    label_to_name_no_negative = label_to_name.copy()
    label_to_name_no_negative.pop(0)
    for key in label_to_name_no_negative.keys():
        pred_counts[key] = pred_counts.get(key, 0)
    pred_counts = np.asarray(
        list(map(lambda x: x[1], sorted(pred_counts.items(), key=lambda elem: elem[0])))
    )
    if return_probabilities:
        return pred_counts, y_pred_proba
    else:
        return pred_counts, None


def score(
    model,
    data_folder,
    directory="opm_test",
    labels_file="test_opm_labels.csv",
    return_counts=True,
    verbose=False,
):
    """
    Compute the MAE on the test OPMs dataset
    :param model: Keras model: already trained classification model
    :param data_folder: str: data folder containing the main directory
    :param directory: str: directory for test OPMs
    :param labels_file: str: labels filename, must be inside the directory
    :param return_counts: boolean, whether or not to return real and predicted counts
    :param verbose: boolean, whether or not to display each OPM filename being processed
    :return: (
        Optional Pandas DataFrame: real counts, Optional Pandas DataFrame: predicted counts, list: computed MAE scores
    )
    """
    labels_blob = join(data_folder, directory, labels_file)
    images_directory = join(data_folder, directory)
    df_labels = pd.read_csv(labels_blob, sep=";").fillna(0).set_index("filename")
    all_pred_counts, all_true_counts, all_mae, filenames = [], [], [], []
    for filename in os.listdir(images_directory):
        if not filename.endswith(".tif"):
            continue
        if verbose:
            print(filename)
        image = imread(join(images_directory, filename))
        chunks = get_chunks(image, input_dim).astype("float32")
        pred_counts, _ = inference(model, chunks, return_probabilities=False)
        true_counts = df_labels.loc[filename.split("/")[-1]].values
        all_pred_counts.append(pred_counts)
        all_true_counts.append(true_counts)
        all_mae.append(mae(true_counts, pred_counts))
        filenames.append(filename)
    if return_counts:
        classes = list(label_to_name.values())[1:]
        all_pred_counts = pd.DataFrame(
            all_pred_counts, columns=classes, index=filenames
        )
        all_true_counts = pd.DataFrame(
            all_true_counts, columns=classes, index=filenames
        )
        return all_true_counts, all_pred_counts, all_mae
    else:
        return None, None, all_mae
