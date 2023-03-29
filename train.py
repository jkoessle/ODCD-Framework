import tensorflow as tf
import numpy as np
import os
import utils.config as cfg
import utils.utilities as utils
import cnn_approach.xai as xai
import cnn_approach.preprocessing_pipeline as pp
import cnn_approach.cnn_module as cnn
from predict import predict
from tf_keras_vis.utils.scores import CategoricalScore


if __name__ == "__main__":

    if cfg.PREPROCESS:
        pp.preprocessing_pipeline(cfg.N_WINDOWS, p_mode="train")
    else:
        cfg.DEFAULT_DATA_DIR = cfg.INTERIM_DATA_DIR

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.DEFAULT_DATA_DIR, subset="both", image_size=cfg.IMAGE_SIZE,
        seed=42, validation_split=0.2, color_mode="rgb")

    date = utils.get_timestamp()
    out_path = utils.create_output_directory(date)

    callbacks = []

    if cfg.CHECKPOINTS:
        checkpoints_dir = os.path.join(out_path, "checkpoints",
                                       f"best_{cfg.MODEL_SELECTION}_{date}.h5")
        checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir,
                                                         monitor="val_accuracy",
                                                         save_best_only=True)
        callbacks.append(checkpoints)
        cfg.MODEL_PATH = checkpoints_dir

    if cfg.EARLY_STOPPING:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                          min_delta=0.001,
                                                          patience=10)
        callbacks.append(early_stopping)

    if cfg.TENSORBOARD:
        log_dir = f"tensorboard_logs/tb_log_{date}_{cfg.MODEL_SELECTION}"
        tf_board = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        callbacks.append(tf_board)

    if cfg.NEW_MODEL:
        model = cnn.cnn_model(model_selection=cfg.MODEL_SELECTION,
                              targetsize=cfg.TARGETSIZE, pretrained=cfg.PRETRAINED,
                              fc_layer=cfg.FC_LAYER, n_classes=cfg.N_CLASSES,
                              dropout=cfg.DROPOUT, agg_layer=cfg.AGG_LAYER,
                              l_r=cfg.L_R, optimizer=cfg.OPTIMIZER)
    else:
        model = tf.keras.models.load_model(cfg.MODEL_PATH)

    if cfg.TRAIN_MODEL:
        model.fit(train_ds, epochs=cfg.EPOCHS, validation_data=val_ds,
                  callbacks=callbacks)

    if cfg.SAVE_MODEL:
        model.save(os.path.join(out_path, f"{cfg.MODEL_SELECTION}_{date}.h5"))

    if cfg.XAI_VIS:
        labels = val_ds.class_names

        images = []
        
        if cfg.CHECKPOINTS and cfg.EARLY_STOPPING:
            best_model = tf.keras.models.load_model(cfg.MODEL_PATH)
        else:
            best_model = model

        for i in range(len(labels)):
            filtered_ds = val_ds.filter(
                lambda _, tmp: tf.math.equal(tmp[0], i))
            for image, label in filtered_ds.take(1):
                images.append(image[0].numpy().astype('uint8'))

        assert len(images) == len(labels), "Did not find one image per label"

        images = np.asarray(images)

        score = CategoricalScore([i for i in range(len(labels))])

        preprocess_images = xai.preprocess_model_input(
            cfg.MODEL_SELECTION, images)

        xai.smooth_grad(best_model, score, preprocess_images, labels, out_path)

        xai.grad_cam(best_model, score, images, preprocess_images, labels, out_path)

        xai.fast_score_cam(best_model, score, images,
                           preprocess_images, labels, out_path)
        
    if cfg.PREDICT:
        predict(best_model, out_path)
