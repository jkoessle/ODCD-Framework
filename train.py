import tensorflow as tf
from cnn_approach.preprocessing_pipeline import preprocessing_pipeline
from cnn_approach.cnn_module import cnn_model
from utils.config import DEFAULT_DATA_DIR, MODEL_SELECTION, L_R
from utils.config import AGG_LAYER, FC_LAYER, OPTIMIZER, N_WINDOWS, TARGETSIZE
from utils.config import DROPOUT, N_CLASSES, PRETRAINED, EPOCHS


if __name__ == "__main__":

    data = preprocessing_pipeline(DEFAULT_DATA_DIR, N_WINDOWS)

    size = (150, 150)

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        DEFAULT_DATA_DIR, subset="both", image_size=size, 
        seed=42, validation_split=0.8, color_mode="rgb")

    # train_ds, val_ds = np.split(data, [int(0.8 * len(data))])

    # size = (150, 150)

    # train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    # val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    # batch_size = 32

    # train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    # val_ds = val_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    model = cnn_model(model_selection=MODEL_SELECTION, targetsize=TARGETSIZE,
                      pretrained=PRETRAINED, fc_layer=FC_LAYER, n_classes=N_CLASSES,
                      dropout=DROPOUT, agg_layer=AGG_LAYER, l_r=L_R, 
                      optimizer=OPTIMIZER)

    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
