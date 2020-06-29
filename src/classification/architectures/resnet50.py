"""
Customizable Resnet50 architecture in TensorFlow/Keras with pretrained Imagenet weights
This implementation uses the model subclassing way
"""

from typing import Tuple
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
    Flatten,
)
from tensorflow.keras.optimizers import Adam


class InnerDenseBlock(Layer):
    def __init__(
        self, hidden_dim=256, activation="elu", dropout=0.25, batch_normalization=True
    ):
        super(InnerDenseBlock, self).__init__()
        self.hidden_dense = Dense(hidden_dim, activation=activation)
        self.dropout_layer = Dropout(dropout) if dropout else None
        self.batch_normalization_layer = (
            BatchNormalization() if batch_normalization else None
        )

    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.hidden_dense(outputs)
        if self.dropout_layer is not None:
            outputs = self.dropout_layer(outputs)
        if self.batch_normalization_layer is not None:
            outputs = self.batch_normalization_layer(outputs, training=training)
        return outputs


class ResNet50Wrapper(Model):
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        output_dim: int,
        hidden_dim_begin=256,
        hidden_dim_min=128,
        freezed_conv_layers=15,
        activation="elu",
        batch_normalization=True,
        dropout=True,
        dropout_begin=0.2,
        dropout_max=0.5,
        final_average_pooling=False,
        depth=0,
        learning_rate=1e-4,
    ):
        super(ResNet50Wrapper, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backbone = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_dim,
            classes=output_dim,
        )
        # Batch-norm layers have to be retrained since the dataset is not the same
        for index, layer in enumerate(self.backbone.layers):
            if index < freezed_conv_layers and not isinstance(
                layer, BatchNormalization
            ):
                layer.trainable = False
            else:
                layer.trainable = True
        self.global_average_pooling = (
            GlobalAveragePooling2D() if final_average_pooling else None
        )
        self.flat = Flatten()
        self.inner_dense_blocks = []
        hidden_dim_current, dropout_current = hidden_dim_begin, dropout_begin
        for depth_i in range(depth):
            if not dropout:
                dropout_current = 0.0
            self.inner_dense_blocks.append(
                InnerDenseBlock(
                    hidden_dim_current,
                    activation,
                    dropout=dropout_current,
                    batch_normalization=batch_normalization,
                )
            )
            hidden_dim_current = max(hidden_dim_current // 2, hidden_dim_min)
            dropout_current = min(dropout_current + 0.1, dropout_max)
        self.dropout = Dropout(dropout_max) if dropout else None
        self.finale_dense = Dense(output_dim, activation="softmax")
        self.optimizer = Adam(learning_rate=learning_rate)
        self.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def call(self, inputs, training=None, mask=None):
        outputs = inputs
        outputs = self.backbone(outputs)
        if self.global_average_pooling is not None:
            outputs = self.global_average_pooling(outputs)
        else:
            outputs = self.flat(outputs)
        for inner_dense_block in self.inner_dense_blocks:
            outputs = inner_dense_block(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        outputs = self.finale_dense(outputs)
        return outputs
