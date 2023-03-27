import tensorflow as tf
from cnn_approach.preprocessing_pipeline import preprocessing_pipeline
from cnn_approach.cnn_module import cnn_model
import utils.config as config
import utils.utilities as utils
# import tensorflow_model_analysis as tfma
# from google.protobuf import text_format

if __name__ == "__main__":

    data = preprocessing_pipeline(
        config.N_WINDOWS, p_mode="train")

    size = (150, 150)

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DEFAULT_DATA_DIR, subset="both", image_size=size,
        seed=42, validation_split=0.2, color_mode="rgb")
    
    date = utils.get_timestamp()

    checkpoints_dir = "checkpoints/" + date
    checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir,
                                                     monitor="val_accuracy",
                                                     save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                      min_delta=0.01,
                                                      patience=7)

    log_dir = "tensorboard_logs/tb_log_" + date
    tf_board = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    model = cnn_model(model_selection=config.MODEL_SELECTION,
                      targetsize=config.TARGETSIZE, pretrained=config.PRETRAINED,
                      fc_layer=config.FC_LAYER, n_classes=config.N_CLASSES,
                      dropout=config.DROPOUT, agg_layer=config.AGG_LAYER,
                      l_r=config.L_R, optimizer=config.OPTIMIZER)

    model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds,
              callbacks=[tf_board, early_stopping])
    
    model.save(f"{config.MODEL_SELECTION}_{date}.h5")

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
