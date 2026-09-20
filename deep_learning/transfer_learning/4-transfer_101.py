#!/usr/bin/env python3
"""Train a classifier on the Caltech-101 dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras as K


def train_transfer_model():
    """Build, train, fine-tune, and save a Caltech-101 classifier."""
    image_size = (224, 224)
    batch_size = 32
    num_classes = 102

    # Load the Caltech-101 training and validation datasets.
    train_data, validation_data = tfds.load(
        "caltech101",
        split=["train", "test"],
        as_supervised=True,
    )

    # Resize and cast each image.
    def prepare_image(image, label):
        """Resize an image and convert it to float32."""
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, label

    # Prepare the training dataset.
    train_data = (
        train_data
        .shuffle(3060, seed=42)
        .map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Prepare the validation dataset.
    validation_data = (
        validation_data
        .map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build the image augmentation pipeline.
    augmentation = K.Sequential([
        K.layers.RandomFlip(
            "horizontal",
            seed=42,
        ),
        K.layers.RandomRotation(
            0.15,
            seed=42,
        ),
        K.layers.RandomZoom(
            0.15,
            seed=42,
        ),
        K.layers.RandomContrast(
            0.1,
            seed=42,
        ),
    ])

    # Load the pretrained MobileNetV2 convolutional base.
    base_model = K.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze the pretrained model for the first training phase.
    base_model.trainable = False

    # Build the transfer-learning model.
    inputs = K.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = K.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(128, activation="relu")(x)
    outputs = K.layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    # Train only the custom classification head.
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=15,
    )

    # Unfreeze the top layers of MobileNetV2 for fine-tuning.
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep batch normalization layers frozen during fine-tuning.
    for layer in base_model.layers[-30:]:
        if isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = False

    # Recompile with a much smaller learning rate.
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fine-tune the upper layers of the pretrained network.
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=15,
    )

    # Save the final trained classifier.
    model.save("caltech101_model.h5")
    