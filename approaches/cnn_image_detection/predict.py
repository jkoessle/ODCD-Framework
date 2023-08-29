import os
import tensorflow as tf
import numpy as np
import utils.config as cfg
from cnn_approach.datasets import create_multilabel_dataset, get_test_data
from utils.evaluate import plot_classification_report, plot_confusion_matrix
from utils.evaluate import plot_multilabel_confusion_matrix


def predict(model, out_path):

    if cfg.MULTILABEL:
        test_ds = create_multilabel_dataset(cfg.TEST_DATA_DIR, training=False)
        logits = model.predict(test_ds, verbose=2)
        
        labels = cfg.DRIFT_TYPES
        labels_idx = [i for i in range(len(labels))]
        
        y_pred = tf.nn.sigmoid(logits)
        # y_hat = y_pred[y_pred > 0.5]
        y_hat = y_pred.numpy().round()
    else:
        test_ds = get_test_data()
        
        logits = model.predict(test_ds, verbose=2)
        
        labels = test_ds.class_names
        labels_idx = [i for i in range(len(labels))]
        
        y_pred = tf.nn.softmax(logits)
        y_hat = np.argmax(y_pred, axis=1)
        
    # get images and labels of test set
    # images = np.concatenate([x for x, y in test_ds], axis=0)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    
    if cfg.MULTILABEL:
        plot_multilabel_confusion_matrix(y_true, y_hat, labels_idx, labels, out_path)
    else:
        plot_confusion_matrix(y_true, y_hat, labels_idx, labels, out_path)
        
    plot_classification_report(y_true, y_hat, labels_idx, labels, out_path)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    model = tf.keras.models.load_model(cfg.MODEL_PATH)
    path = os.getcwd()

    predict(model, path)
