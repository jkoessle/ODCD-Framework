import os
import tensorflow as tf
import utils.config as cfg
import pandas as pd


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


def parse_image_labels(filename, label):

    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, cfg.IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, label


def get_multilabel_information(dir):
    # load generated label information
    csv_path = os.path.join(dir, "labels.csv")
    labels_df = pd.read_csv(csv_path, sep=",")

    # get list of filenames and their paths
    filenames = labels_df["filenames"].values
    filenames = list(map(lambda x: os.path.join(dir, x), filenames))

    # get corresponding labels
    labels = labels_df[cfg.DRIFT_TYPES].values

    return filenames, labels


def create_multilabel_dataset(dir, training=True):
    # get filenames and labels
    filenames, labels = get_multilabel_information(dir)

    # create dataset and map labels to images
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_image_labels,
                          num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # cache and shuffle if training mode
        dataset = dataset.cache().shuffle(buffer_size=cfg.SHUFFLE_BUFFER_SIZE)

    # batch and prefetch data
    dataset = dataset.batch(cfg.BATCH_SIZE).prefetch(
        buffer_size=tf.data.AUTOTUNE)

    return dataset
