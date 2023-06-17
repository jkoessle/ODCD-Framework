import os
import glob
import tensorflow as tf
from . import config as cfg
import cnn_image_detection.utils.utilities as utils
import official.vision.data.create_coco_tf_record as coco


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
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
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
        "image_id": int64_feature(example["image_id"]),
        "iscrowd": int64_feature(example["iscrowd"]),
        "ignore": int64_feature(example["ignore"]),
        #TODO: fix segmentation list
        "segmentation": float_feature_list(example["segmentation"]),
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
        "image_id": tf.io.FixedLenFeature([], tf.int64),
        "iscrowd": tf.io.FixedLenFeature([], tf.int64),
        "ignore": tf.io.FixedLenFeature([], tf.int64),
        "segmentation": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example


def generate_tfr_data(img_dir, annotations):
    num_samples = len(glob.glob(os.path.join(img_dir,"*.jpg")))
    
    num_tfrecords = len(annotations) // num_samples
    if len(annotations) % num_samples:
        num_tfrecords += 1

    tfr_dir = os.path.join(cfg.TFR_RECORDS_DIR, f"tfr_data_{utils.get_timestamp()}")

    if not os.path.exists(tfr_dir):
        os.makedirs(tfr_dir) 
        
    annotations = annotations["annotations"]

    for tfrec_num in range(num_tfrecords):

        samples = annotations[(tfrec_num * num_samples)
                               : ((tfrec_num + 1) * num_samples)]

        with tf.io.TFRecordWriter(
            tfr_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for sample in samples:
                image_path = f"{img_dir}/{sample['image_id']}.jpg"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())


def prepare_sample(features):
    image = tf.image.resize(features["image"], size=cfg.IMAGE_SIZE)
    return image, features["category_id"], features["bbox"], features["image_id"]


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


def generate_tfr_data_from_coco_annotations(img_dir):
    
    annotations = os.path.join(img_dir, "annotations.json")

    tfr_dir = os.path.join(cfg.TFR_RECORDS_DIR,
                           f"tfr_data_{utils.get_timestamp()}")

    if not os.path.exists(tfr_dir):
        os.makedirs(tfr_dir)

    coco._create_tf_record_from_coco_annotations(images_info_file=annotations,
                                                 image_dirs=img_dir,
                                                 output_path=tfr_dir,
                                                 num_shards=cfg.N_SHARDS)
