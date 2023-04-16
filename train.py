import tensorflow as tf
import numpy as np
import os
import argparse
import utils.config as cfg
import utils.utilities as utils
import cnn_approach.xai as xai
import cnn_approach.preprocessing_pipeline as pp
import cnn_approach.cnn_module as cnn
from cnn_approach.datasets import get_train_split
from cnn_approach.datasets import create_multilabel_dataset
from predict import predict
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore

#TODO add more expresive statements during execution

if __name__ == "__main__":

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--gpu_devices", dest="gpu_devices", 
                        help="Specify which CUDA devices to use.",
                        default="",
                        type=str)
    args = parser.parse_args()

    # set cuda devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    date = utils.get_timestamp()
    out_path = utils.create_output_directory(date)

    if cfg.PREPROCESS:
        pp.preprocessing_pipeline(cfg.N_WINDOWS, p_mode="train")
    else:
        cfg.DEFAULT_DATA_DIR = cfg.INTERIM_DATA_DIR

    if cfg.MULTILABEL:
        train_ds = create_multilabel_dataset(cfg.TRAIN_DATA_DIR)
        val_ds = create_multilabel_dataset(cfg.EVAL_DATA_DIR)
        monitor = "val_binary_accuracy"
    else:
        train_ds, val_ds = get_train_split()
        monitor = "val_sparse_categorical_accuracy"

    callbacks = []

    if cfg.NEW_MODEL:
        model = cnn.cnn_model(model_selection=cfg.MODEL_SELECTION,
                              targetsize=cfg.TARGETSIZE, pretrained=cfg.PRETRAINED,
                              fc_layer=cfg.FC_LAYER, n_classes=cfg.N_CLASSES,
                              dropout=cfg.DROPOUT, agg_layer=cfg.AGG_LAYER,
                              l_r=cfg.L_R, optimizer=cfg.OPTIMIZER)
    else:
        model = tf.keras.models.load_model(cfg.MODEL_PATH)

    if cfg.TRAIN_MODEL:
        if cfg.CHECKPOINTS:
            checkpoints_dir = os.path.join(out_path, "checkpoints",
                                           f"best_{cfg.MODEL_SELECTION}_{date}.h5")
            checkpoints = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoints_dir,
                monitor=monitor,
                save_best_only=True,
                verbose=1)
            callbacks.append(checkpoints)
            cfg.MODEL_PATH = checkpoints_dir

        if cfg.EARLY_STOPPING:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=0.001,
                patience=cfg.EARLY_STOPPING_PATIENCE,
                verbose=1)
            callbacks.append(early_stopping)

        if cfg.TENSORBOARD:
            log_dir = f"tensorboard_logs/tb_log_{date}_{cfg.MODEL_SELECTION}"
            tf_board = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            callbacks.append(tf_board)
            model.fit(train_ds, epochs=cfg.EPOCHS, validation_data=val_ds,
                      callbacks=callbacks, verbose=1)

    if cfg.SAVE_MODEL:
        model.save(os.path.join(out_path, f"{cfg.MODEL_SELECTION}_{date}.h5"))

    if cfg.XAI_VIS:
        labels = cfg.DRIFT_TYPES

        images = []

        if cfg.CHECKPOINTS:
            best_model = tf.keras.models.load_model(cfg.MODEL_PATH)
        else:
            best_model = model

        if cfg.MULTILABEL:
            images, image_labels = utils.get_multilabel_samples(val_ds)
            
            score = BinaryScore(0.0)
        else:
            for i in range(len(labels)):
                filtered_ds = val_ds.filter(
                    lambda _, tmp: tf.math.equal(tmp[0], i))
                for image, label in filtered_ds.take(1):
                    images.append(image[0].numpy().astype('uint8'))

            assert len(images) == len(
                labels), "Did not find one image per label"
            
            images = np.asarray(images)

            score = CategoricalScore([i for i in range(len(labels))])

        preprocess_images = xai.preprocess_model_input(
            cfg.MODEL_SELECTION, images)

        xai.smooth_grad(best_model, score,
                        preprocess_images, labels, out_path)

        xai.grad_cam(best_model, score, images,
                        preprocess_images, labels, out_path)

        xai.fast_score_cam(best_model, score, images,
                            preprocess_images, labels, out_path)

    if cfg.PREDICT:
        predict(best_model, out_path)
