"""
Endpoint script for classification model training
Should not be used directly but as entry script in src/experiment.py
"""

import argparse
import warnings
import os
import pickle
import time
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from azureml.core import Run

from classification.architectures.resnet50 import ResNet50Wrapper
from classification.architectures.callbacks import CyclicLR
from config import GeneralConfig, ModelConfig, PathsConfig

from plot import plot_confusion_matrix


class LogRunMetrics(Callback):
    def on_epoch_end(self, epoch, log=None):
        if "val_loss" in log and "val_accuracy" in log:
            run.log("Loss", log["val_loss"])
            run.log("Accuracy", log["val_accuracy"])


def initialize_model(args, output_dim) -> ResNet50Wrapper:
    if GeneralConfig.architecture_type == "PretrainedResNet50":
        model_class = ResNet50Wrapper
        hyperparams = ModelConfig.pretrained_resnet50_hyperparams.copy()
    else:
        raise NotImplementedError
    dict_args = vars(args)
    for key, value in hyperparams.items():
        hyperparams[key] = (
            dict_args.get(key) if dict_args.get(key) is not None else value
        )
    hyperparams["input_dim"] = (hyperparams["input_dim"], hyperparams["input_dim"], 3)
    hyperparams["output_dim"] = output_dim
    return model_class(**hyperparams)


def initialize_callbacks():
    azureml_logger = LogRunMetrics()
    early_stopper = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=ModelConfig.patience,
        restore_best_weights=True,
        verbose=GeneralConfig.verbose,
    )
    clr = CyclicLR(
        base_lr=learning_rate / 10, max_lr=learning_rate,
        step_size=step_size, verbose=1,
        monitor="val_loss",
        reduce_on_plateau=None
    )
    callbacks = azureml_logger, early_stopper, clr
    return callbacks


def get_data_generator(augmented: bool):
    if augmented:
        generator_schema = ImageDataGenerator(
            rotation_range=ModelConfig.rotation_range,
            zoom_range=ModelConfig.zoom_range,
            width_shift_range=ModelConfig.width_shift_range,
            height_shift_range=ModelConfig.height_shift_range,
            horizontal_flip=ModelConfig.horizontal_flip,
            vertical_flip=ModelConfig.vertical_flip,
            featurewise_center=ModelConfig.featurewise_center,
            featurewise_std_normalization=ModelConfig.featurewise_std_normalization,
            samplewise_center=ModelConfig.samplewise_center,
            samplewise_std_normalization=ModelConfig.samplewise_std_normalization,
            rescale=ModelConfig.rescale,
            preprocessing_function=preprocess_input
        )
    else:
        generator_schema = ImageDataGenerator(
            featurewise_center=ModelConfig.featurewise_center,
            featurewise_std_normalization=ModelConfig.featurewise_std_normalization,
            samplewise_center=ModelConfig.samplewise_center,
            samplewise_std_normalization=ModelConfig.samplewise_std_normalization,
            rescale=ModelConfig.rescale,
            preprocessing_function=preprocess_input
        )
    return generator_schema


def tta_inference(model: ResNet50Wrapper, data_folder: str, input_dim: int, passes: int) -> np.ndarray:
    tta_predictions = []
    samples, _ = next(
        ImageDataGenerator().flow_from_directory(
            directory=os.path.join(data_folder, PathsConfig.data_train),
            batch_size=ModelConfig.normalization_samples,
            shuffle=True,
            target_size=(input_dim, input_dim),
            class_mode='categorical',
        )
    )
    test_generator_schema = get_data_generator(augmented=True)
    test_generator_schema.fit(samples)
    for iteration in range(passes):
        test_generator = test_generator_schema.flow_from_directory(
            os.path.join(data_folder, PathsConfig.data_test),
            batch_size=ModelConfig.batch_size,
            shuffle=False,
            target_size=(input_dim, input_dim),
            class_mode='categorical'
        )
        tta_predictions.append(model.predict(test_generator))
    return np.mean(tta_predictions, axis=0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder")
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--filters-dim-begin", type=int, default=None)
    parser.add_argument("--filters-dim-max", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--batch-normalization", type=bool, default=None)
    parser.add_argument("--dropout", type=bool, default=None)
    parser.add_argument("--dropout-begin", type=float, default=None)
    parser.add_argument("--dropout-max", type=float, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--residual-connexion", type=bool, default=None)
    parser.add_argument("--max-pooling", type=bool, default=None)
    parser.add_argument("--final-average-pooling", type=bool, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()
    data_folder = args.data_folder

    input_dim = args.input_dim or ModelConfig.pretrained_resnet50_hyperparams["input_dim"]
    learning_rate = args.learning_rate or ModelConfig.pretrained_resnet50_hyperparams["learning_rate"]

    print("Initializing generators...")
    augmented_generator_schema, generator_schema = \
        get_data_generator(augmented=True), get_data_generator(augmented=False)
    train_generator_schema = augmented_generator_schema if ModelConfig.data_augmentation else generator_schema
    test_generator_schema = generator_schema

    samples, _ = next(
        ImageDataGenerator().flow_from_directory(
            directory=os.path.join(data_folder, PathsConfig.data_train),
            batch_size=ModelConfig.normalization_samples,
            shuffle=True,
            target_size=(input_dim, input_dim),
            class_mode='categorical',
        )
    )
    train_generator_schema.fit(samples)
    test_generator_schema.fit(samples)

    os.makedirs(PathsConfig.generators_directory, exist_ok=True)

    with open(PathsConfig.augmented_image_generator_path, 'wb') as fp:
        pickle.dump(augmented_generator_schema, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PathsConfig.image_generator_path, 'wb') as fp:
        pickle.dump(generator_schema, fp, protocol=pickle.HIGHEST_PROTOCOL)

    train_generator = train_generator_schema.flow_from_directory(
        directory=os.path.join(data_folder, PathsConfig.data_train),
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        target_size=(input_dim, input_dim),
        class_mode='categorical'
    )

    test_generator = test_generator_schema.flow_from_directory(
        os.path.join(data_folder, PathsConfig.data_test),
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        target_size=(input_dim, input_dim),
        class_mode='categorical'
    )

    num_samples = train_generator.n
    steps_per_epoch = np.ceil(num_samples / ModelConfig.batch_size).astype(int)
    step_size = np.ceil(steps_per_epoch / 2).astype(int)

    run = Run.get_context()
    callbacks = initialize_callbacks()
    model = initialize_model(args, train_generator.num_classes)

    print("Training model...")
    time_anchor = time.time()
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        verbose=GeneralConfig.verbose,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=test_generator.samples // test_generator.batch_size,
        epochs=ModelConfig.epochs,
        callbacks=callbacks
    )
    training_time = time.time() - time_anchor
    print("Model trained")
    print("Inference on test data...")
    os.makedirs(PathsConfig.model_directory, exist_ok=True)
    os.makedirs(PathsConfig.predictions_directory, exist_ok=True)
    model.save(PathsConfig.model_directory)
    try:
        model = load_model(PathsConfig.model_directory)
    except OSError:
        print("Model not saved properly!")
    test_generator_inference = test_generator_schema.flow_from_directory(
        os.path.join(data_folder, PathsConfig.data_test),
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        target_size=(input_dim, input_dim),
        class_mode='categorical'
    )
    time_anchor = time.time()
    y_pred_proba = model.predict(test_generator_inference)
    y_pred = np.argmax(y_pred_proba, axis=1)
    per_sample_inference_time = (time.time() - time_anchor) / test_generator_inference.n

    y_true = test_generator_inference.labels

    time_anchor = time.time()
    averaged_tta_predictions_proba = tta_inference(
        model=model,
        data_folder=data_folder,
        input_dim=input_dim,
        passes=ModelConfig.tta_augmentation_passes
    )
    averaged_tta_predictions = np.argmax(averaged_tta_predictions_proba, axis=1)
    per_sample_tta_inference_time = (time.time() - time_anchor) / test_generator_inference.n

    run.log("Final test loss", log_loss(y_true, y_pred_proba))
    run.log("Final test accuracy", accuracy_score(y_true, y_pred))
    run.log("Final TTA test accuracy", accuracy_score(y_true, averaged_tta_predictions))
    run.log("Training time", training_time)
    run.log("Per sample inference time", per_sample_inference_time)
    run.log("Per sample TTA inference time", per_sample_tta_inference_time)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        y_true,
        y_pred,
        np.asarray(list(train_generator.class_indices.keys())),
        path=PathsConfig.confusion_matrix_path,
    )
    run.log_image(name="Confusion matrix", path=PathsConfig.confusion_matrix_path)
