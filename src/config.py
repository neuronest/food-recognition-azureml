import os


class GeneralConfig:
    # General
    validation_split = 0.2
    seed = 42
    verbose = True
    architecture_type = "PretrainedResNet50"
    # Data image extension
    image_extension = ".jpg"
    pip_packages = [
        "tensorflow==2.2",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-image",
        "scikit-learn",
        "opencv-python",
        "tqdm",
    ]
    # Enable hyperdrive experiments
    hyperdrive = True


class ModelConfig:
    # General model hyperparameters
    batch_size = 64
    epochs = 30
    # Early stopping
    patience = 5
    # Normalization samples
    normalization_samples = 1000
    # Data augmentation
    data_augmentation = True
    rotation_range = 40
    zoom_range = 0.2
    width_shift_range = 0.2
    height_shift_range = 0.2
    horizontal_flip = True
    vertical_flip = False
    featurewise_center = True
    featurewise_std_normalization = True
    samplewise_center = False
    samplewise_std_normalization = False
    rescale = None
    # TTA augmentation passes
    tta_augmentation_passes = 10
    # Pretrained ResNet50 hyperparameters
    pretrained_resnet50_hyperparams = {
        "input_dim": 224,
        "learning_rate": 1e-4,
        "hidden_dim_begin": 256,
        "hidden_dim_min": 128,
        "freezed_conv_layers": 15,
        "activation": "elu",
        "batch_normalization": True,
        "dropout": True,
        "dropout_begin": 0.2,
        "dropout_max": 0.5,
        "final_average_pooling": False,
        "depth": 2,
    }


class HyperdriveConfig:
    # Pretrained ResNet50 hyperparameters
    pretrained_resnet50_hyperparams_space = {
        "--input-dim": [112, 224],
        "--learning-rate": [1e-4],
        "--hidden-dim-begin": [256],
        "--hidden-dim-min": [128],
        "--freezed-conv-layers": [5, 15, 30],
        "--activation": ["elu"],
        "--batch-normalization": [True],
        "--dropout": [True],
        "--dropout-begin": [0.2],
        "--dropout-max": [0.5],
        "--final-average-pooling": [False, True],
        "--depth": [0, 1, 2],
    }
    evaluation_interval = 2
    slack_factor = 0.1
    max_total_runs = 100
    max_concurrent_runs = 1


class PathsConfig:
    # Data paths
    entry_script = "classification/training.py"
    data_train = "train"
    data_test = "test"
    classes = "classes.txt"
    # Outputs paths
    outputs_directory = "outputs"
    generators_directory = os.path.join(outputs_directory, "generators")
    image_generator_path = os.path.join(generators_directory, "image_generator.pkl")
    augmented_image_generator_path = os.path.join(
        generators_directory, "augmented_image_generator.pkl"
    )
    predictions_directory = os.path.join(outputs_directory, "predictions")
    model_directory = os.path.join(outputs_directory, "model")
    confusion_matrix_path = "confusion_matrix.jpg"
