import tensorflow as tf
import utils.config as cfg 
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
# from keras.layers import GlobalAveragePooling2D, Dropout
# from keras.models import Model
# from keras.optimizers import Adam, Adagrad, Adadelta, SGD, RMSprop


def augmentation(input):
    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    #     tf.keras.layers.RandomRotation(0.2),
    # ])
    
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42)(input)
    data_augmentation = tf.keras.layers.RandomRotation(0.2, seed=42)(x)
    
    return data_augmentation


def baseline_cnn(inputs):

    # First convolutional layer with 32 filters and 3x3 kernel
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                   activation='relu', padding='same')(inputs)
    # Second convolutional layer with 64 filters and 3x3 kernel
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv1)
    # Third convolutional layer with 64 filters and 3x3 kernel
    conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv2)

    # Max pooling layer with 2x2 pool size
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Fourth convolutional layer with 128 filters and 3x3 kernel
    conv4 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3),
                   activation='relu', padding='same')(pool1)
    # Fifth convolutional layer with 128 filters and 3x3 kernel
    conv5 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv4)
    # Sixth convolutional layer with 128 filters and 3x3 kernel
    conv6 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv5)

    # Max pooling layer with 2x2 pool size
    baselin_cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv6)

    return baselin_cnn


def get_optimizer(optimizer, l_r):
    if optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=l_r)
    elif optimizer == "adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=l_r)
    elif optimizer == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=l_r)
    elif optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=l_r)
    elif optimizer == "rms_p":
        return tf.keras.optimizers.RMSprop(learning_rate=l_r)


def pretrained_cnn(model, input_shape, pretrained, input, weights="imagenet"):
    if not pretrained:
        weights=None
    
    if model == "resnet":
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=input_shape,
            weights=weights, input_tensor=input)
    elif model == "inception":
        base_model = tf.keras.applications.InceptionV3(
            include_top=False, input_shape=input_shape,
            weights=weights, input_tensor=input)
    elif model == "inc_res":
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top=False, input_shape=input_shape,
            weights=weights, input_tensor=input)
    elif model == "resnet_rs":
        base_model = tf.keras.applications.ResNetRS50(
            include_top=False, input_shape=input_shape,
            weights=weights, input_tensor=input)
    elif model == "xception":
        base_model = tf.keras.applications.Xception(
            include_top=False, input_shape=input_shape,
            weights=weights, input_tensor=input)
    if pretrained:
        base_model.trainable = False
        
    return base_model.output


def cnn_model(model_selection, targetsize, pretrained, fc_layer, n_classes, 
              dropout=0.25, agg_layer="average", l_r=3e-4, optimizer="adam"):
    
    # create input
    input_shape = (targetsize, targetsize, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    if cfg.AUGMENTATION:
        augment = augmentation(inputs)
    else:
        augment = inputs

    # select model
    if model_selection != "baseline":
        model_arc = pretrained_cnn(model=model_selection, 
                                   input_shape=input_shape, 
                                   pretrained=pretrained,
                                   input=augment)
    else:
        model_arc = baseline_cnn(augment)
        

    # add aggregation layer
    if agg_layer == "flatten":    
        x = tf.keras.layers.Flatten()(model_arc)
    elif agg_layer == "average":
        x = tf.keras.layers.GlobalAveragePooling2D()(model_arc)
        x = tf.keras.layers.Dropout(dropout)(x)
    
    # add fully connected layer
    for layer in fc_layer:
        x = tf.keras.layers.Dense(layer, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    # add output layer
    outputs = tf.keras.layers.Dense(n_classes)(x)

    # create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_model")
    
    # get summary
    model.summary()

    optim = get_optimizer(optimizer=optimizer, l_r=l_r)
    
    # compile model
    model.compile(optimizer=optim,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    
    return model
