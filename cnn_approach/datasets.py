import os
import tensorflow as tf
import utils.config as cfg
from pathlib import Path
# from sklearn.preprocessing import MultiLabelBinarizer


def get_train_split():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.DEFAULT_DATA_DIR, subset="both", image_size=cfg.IMAGE_SIZE,
        shuffle=True, seed=42, validation_split=0.2, color_mode="rgb")
    return train_ds, val_ds


def get_test_data():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TEST_DATA_DIR, image_size=cfg.IMAGE_SIZE,
        shuffle=False, color_mode="rgb")
    return test_ds


def get_multilabel_train_data():
    ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TRAIN_DATA_DIR, labels=None, image_size=cfg.IMAGE_SIZE,
        shuffle=False, color_mode="rgb")

    labels = get_multilabels(cfg.TRAIN_DATA_DIR)

    train_ds = tf.data.Dataset.zip(
        (ds, tf.data.Dataset.from_tensor_slices(labels)))

    return train_ds


def get_multilabel_validation_data():
    ds = tf.keras.utils.image_dataset_from_directory(
        cfg.EVAL_DATA_DIR, labels=None, image_size=cfg.IMAGE_SIZE,
        shuffle=False, color_mode="rgb")

    labels = get_multilabels(cfg.EVAL_DATA_DIR)

    val_ds = tf.data.Dataset.zip(
        (ds, tf.data.Dataset.from_tensor_slices(labels)))

    return val_ds


def get_multilabel_test_data():
    ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TEST_DATA_DIR, labels=None, image_size=cfg.IMAGE_SIZE,
        shuffle=False, color_mode="rgb")

    labels = get_multilabels(cfg.TEST_DATA_DIR)

    test_ds = tf.data.Dataset.zip(
        (ds, tf.data.Dataset.from_tensor_slices(labels)))

    return test_ds


def get_multilabels(dir):

    list_of_files = [f for f in os.listdir(dir) if f.endswith(".png")]

    multilabels = []

    # get labels of images based on filename
    for file in list_of_files:
        filename = Path(file).stem
        labels = filename.split("_")[1:]
        labels_idx = tuple([cfg.DRIFT_TYPES.index(elem) for elem in labels])
        multilabels.append(labels_idx)

        # multilabels.append(tuple(labels))

    # mlb = MultiLabelBinarizer(sparse_output=False)
    # one_hot_enc = mlb.fit_transform(multilabels)

    tf_magic = tf.one_hot(tf.ragged.constant(multilabels), cfg.N_CLASSES)

    return tf_magic
