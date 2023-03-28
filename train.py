import tensorflow as tf
from cnn_approach.preprocessing_pipeline import preprocessing_pipeline
from cnn_approach.cnn_module import cnn_model
import utils.config as cfg
import utils.utilities as utils
# import tensorflow_model_analysis as tfma
# from google.protobuf import text_format

if __name__ == "__main__":

    if cfg.PREPROCESS:
        preprocessing_pipeline(cfg.N_WINDOWS, p_mode="train")
    else:
        cfg.DEFAULT_DATA_DIR = cfg.INTERIM_DATA_DIR

    size = (150, 150)

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.DEFAULT_DATA_DIR, subset="both", image_size=size,
        seed=42, validation_split=0.2, color_mode="rgb")

    date = utils.get_timestamp()
    
    callbacks = []

    if cfg.CHECKPOINTS:
        checkpoints_dir = "checkpoints/" + date
        checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir,
                                                        monitor="val_accuracy",
                                                        save_best_only=True)
        callbacks.append(checkpoints)

    if cfg.EARLY_STOPPING:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                      min_delta=0.01,
                                                      patience=7)
        callbacks.append(early_stopping)
        
    if cfg.TENSORBOARD:
        log_dir = "tensorboard_logs/tb_log_" + date
        tf_board = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        callbacks.append(tf_board)

    model = cnn_model(model_selection=cfg.MODEL_SELECTION,
                      targetsize=cfg.TARGETSIZE, pretrained=cfg.PRETRAINED,
                      fc_layer=cfg.FC_LAYER, n_classes=cfg.N_CLASSES,
                      dropout=cfg.DROPOUT, agg_layer=cfg.AGG_LAYER,
                      l_r=cfg.L_R, optimizer=cfg.OPTIMIZER)

    model.fit(train_ds, epochs=cfg.EPOCHS, validation_data=val_ds,
              callbacks=callbacks)

    if cfg.SAVE_MODEL:
        model.save(f"{cfg.MODEL_SELECTION}_{date}.h5")

    # metrics_specs = text_format.Parse("""
    # metrics_specs {
    #     metrics { class_name: "ExampleCount" }
    #     metrics { class_name: "SparseCategoricalCrossentropy" }
    #     metrics { class_name: "SparseCategoricalAccuracy" }
    #     metrics { class_name: "Precision" config: '"top_k": 1' }
    #     metrics { class_name: "Precision" config: '"top_k": 3' }
    #     metrics { class_name: "Recall" config: '"top_k": 1' }
    #     metrics { class_name: "Recall" config: '"top_k": 3' }
    #     metrics { class_name: "MultiClassConfusionMatrixPlot" }
    # }
    # """, tfma.EvalConfig()).metrics_specs
