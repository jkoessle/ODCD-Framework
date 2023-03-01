import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def augmentation():
    pass


def baselin_cnn(inputs):

    # First convolutional layer with 32 filters and 3x3 kernel
    conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu', padding='same')(inputs)
    # Second convolutional layer with 64 filters and 3x3 kernel
    conv2 = Conv2D(64, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv1)
    # Third convolutional layer with 64 filters and 3x3 kernel
    conv3 = Conv2D(64, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv2)

    # Max pooling layer with 2x2 pool size
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Fourth convolutional layer with 128 filters and 3x3 kernel
    conv4 = Conv2D(128, kernel_size=(3, 3),
                   activation='relu', padding='same')(pool1)
    # Fifth convolutional layer with 128 filters and 3x3 kernel
    conv5 = Conv2D(128, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv4)
    # Sixth convolutional layer with 128 filters and 3x3 kernel
    conv6 = Conv2D(128, kernel_size=(3, 3),
                   activation='relu', padding='same')(conv5)

    # Max pooling layer with 2x2 pool size
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # Flatten the output from the convolutional layers
    baselin_cnn = Flatten()(pool2)

    return baselin_cnn


def pretrained_cnn(model, input_tensor, input_shape, weights="imagenet"):

    if model == "resnet":
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_tensor=input_tensor, input_shape=input_shape,
            weights=weights, pooling="max")
    elif model == "inception":
        base_model = tf.keras.applications.InceptionV3(
            include_top=False, input_tensor=input_tensor, input_shape=input_shape,
            weights=weights, pooling="max")
    elif model == "inc_res":
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top=False, input_tensor=input_tensor, input_shape=input_shape,
            weights=weights, pooling="max")
    elif model == "resnet_rs":
        base_model = tf.keras.applications.ResNetRS50(
            include_top=False, input_tensor=input_tensor, input_shape=input_shape,
            weights=weights, pooling="max")
    elif model == "xception":
        base_model = tf.keras.applications.Xception(
            include_top=False, input_tensor=input_tensor, input_shape=input_shape,
            weights=weights, pooling="max")
    return base_model


def cnn_model(model_selection, targetsize, pretrained, n_classes):
    input_shape = (targetsize, targetsize, 1)
    inputs = Input(shape=input_shape)

    if pretrained:
        model_arc = pretrained_cnn(model=model_selection, input_tensor=inputs,
                                   input_shape=input_shape)
    else:
        model_arc = baselin_cnn(inputs)

    fc = Dense(1024, activation="relu")(model_arc)

    outputs = Dense(n_classes, activtion="softmax")(fc)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
