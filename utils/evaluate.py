import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def recall(y_true, y_pred):
    tp = tf.keras.backend.sum(tf.keras.backend.round(
        tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))

    recall = tp / (positives+tf.keras.backend.epsilon())

    return recall


def precision(y_true, y_pred):
    tp = tf.keras.backend.sum(tf.keras.backend.round(
        tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    pred_positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))

    precision = tp / (pred_positives+tf.keras.backend.epsilon())

    return precision


def f1(y_true, y_pred):
    prec, rec = precision(y_true, y_pred), recall(y_true, y_pred)

    return 2*((prec*rec)/(prec+rec+tf.keras.backend.epsilon()))


def plot_classification_report(y_true, y_pred, labels_idx, labels, path):

    c_r = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=labels, output_dict=True)

    # exclude support from clf report with iloc
    clf_r = sns.heatmap(pd.DataFrame(
        c_r).iloc[:-1, :].T, cmap="YlGn", annot=True)
    clf_r_fig = clf_r.get_figure()
    save_path = os.path.join(path, "classification_report.png")
    clf_r_fig.savefig(save_path)
    print(f"Classification report is saved at: {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels_idx, labels, path):

    # create confusion matrix and display
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # render
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xticks(rotation=45)
    disp.plot(ax=ax, cmap="YlGn")
    save_path = os.path.join(path, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion matrix is saved at: {save_path}")
