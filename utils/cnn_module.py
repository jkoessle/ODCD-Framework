import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam, Adagrad, Adadelta, SGD, RMSprop


def augmentation():
    pass


def baseline_cnn(inputs):

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
    baselin_cnn = MaxPooling2D(pool_size=(2, 2))(conv6)

    return baselin_cnn


def get_optimizer(optimizer, l_r):
    if optimizer == "adam":
        return Adam(learning_rate=l_r)
    elif optimizer == "adadelta":
        return Adadelta(learning_rate=l_r)
    elif optimizer == "adagrad":
        return Adagrad(learning_rate=l_r)
    elif optimizer == "sgd":
        return SGD(learning_rate=l_r)
    elif optimizer == "rms_p":
        return RMSprop(learning_rate=l_r)


def pretrained_cnn(model, input_shape, pretrained, weights="imagenet"):
    if not pretrained:
        weights=None
    
    if model == "resnet":
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=input_shape,
            weights=weights)
    elif model == "inception":
        base_model = tf.keras.applications.InceptionV3(
            include_top=False, input_shape=input_shape,
            weights=weights)
    elif model == "inc_res":
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top=False, input_shape=input_shape,
            weights=weights)
    elif model == "resnet_rs":
        base_model = tf.keras.applications.ResNetRS50(
            include_top=False, input_shape=input_shape,
            weights=weights)
    elif model == "xception":
        base_model = tf.keras.applications.Xception(
            include_top=False, input_shape=input_shape,
            weights=weights)
    if pretrained:
        base_model.trainable = False
        
    return base_model


def cnn_model(model_selection, targetsize, pretrained, fc_layer, n_classes, 
              dropout=0.25, agg_layer="average", l_r=3e-4, optimizer="adam"):
    
    # create input
    input_shape = (targetsize, targetsize, 3)
    inputs = Input(shape=input_shape)

    # select model
    if pretrained:
        model_arc = pretrained_cnn(model=model_selection, input_tensor=inputs,
                                   input_shape=input_shape, pretrained=pretrained)
        model_arc = model_arc(inputs, training=False)
    else:
        model_arc = baseline_cnn(inputs)
        

    # add aggregation layer
    if agg_layer == "flatten":    
        x = Flatten()(model_arc)
    elif agg_layer == "average":
        x = GlobalAveragePooling2D()(model_arc)
        x = Dropout(dropout)(x)
    
    # add fully connected layer
    for layer in fc_layer:
        x = Dense(layer, activation="relu")(x)
        x = Dropout(dropout)(x)

    # add output layer
    outputs = Dense(n_classes, activation="softmax")(x)

    # create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # get summary
    model.summary()

    optim = get_optimizer(optimizer=optimizer, l_r=l_r)
    
    # compile model
    model.compile(optimizer=optim,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
