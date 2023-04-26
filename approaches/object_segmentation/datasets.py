import os
import glob
import tensorflow as tf
import cnn_image_detection.utils.config as cfg
import cnn_image_detection.utils.utilities as utils


""" 
Title: Creating TFRecords
Author: Dimitre Oliveira
Date created: 2021/02/27
Last modified: 2021/02/27
Description: Converting data to the TFRecord format.
Availability: 
https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/creating_tfrecords.py
Note: Most of the following functions were adapted from the source code as required.
"""
def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, example):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "area": float_feature(example["area"]),
        "bbox": float_feature_list(example["bbox"]),
        "category_id": int64_feature(example["category_id"]),
        "id": int64_feature(example["id"]),
        "image_id": bytes_feature(example["image_id"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_png(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example


def generate_tfr_data(img_dir, annotations):
    num_samples = len(glob.glob(os.path.join(img_dir,"*.png")))
    
    num_tfrecords = len(annotations) // num_samples
    if len(annotations) % num_samples:
        num_tfrecords += 1

    tfr_dir = os.path.join(cfg.TFR_RECORDS_DIR, f"tfr_data_{utils.get_timestamp()}")

    if not os.path.exists(tfr_dir):
        os.makedirs(tfr_dir) 

    for tfrec_num in range(num_tfrecords):

        samples = annotations[(tfrec_num * num_samples)
                               : ((tfrec_num + 1) * num_samples)]

        with tf.io.TFRecordWriter(
            tfr_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for sample in samples:
                image_path = f"{img_dir}/{sample['image_id']}.png"
                image = tf.io.decode_png(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())


def prepare_sample(features):
    image = tf.image.resize(features["image"], size=cfg.IMAGE_SIZE)
    return image, features["category_id"], features["bbox"]


def get_tfr_dataset(filenames, batch_size=cfg.BATCH_SIZE):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def get_tfr_filenames(dir):
    path = os.path.join(dir, "*.tfrec")
    filenames = tf.io.gfile.glob(path)
    return filenames