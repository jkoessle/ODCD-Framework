import tensorflow as tf
from cnn_approach.preprocessing_pipeline import preprocessing_pipeline
from cnn_approach.cnn_module import cnn_model
import utils.config as config


if __name__ == "__main__":

    data = preprocessing_pipeline(
        config.DEFAULT_TRAIN_DATA_DIR, config.N_WINDOWS)

    size = (150, 150)

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DEFAULT_TRAIN_DATA_DIR, subset="both", image_size=size,
        seed=42, validation_split=0.2, color_mode="rgb")

    # train_ds, val_ds = np.split(data, [int(0.8 * len(data))])

    # size = (150, 150)

    # train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    # val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    # batch_size = 32

    # train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    # val_ds = val_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    model = cnn_model(model_selection=config.MODEL_SELECTION,
                      targetsize=config.TARGETSIZE, pretrained=config.PRETRAINED, 
                      fc_layer=config.FC_LAYER, n_classes=config.N_CLASSES,
                      dropout=config.DROPOUT, agg_layer=config.AGG_LAYER, 
                      l_r=config.L_R, optimizer=config.OPTIMIZER)

    model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds)
