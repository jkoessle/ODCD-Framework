import os
import tensorflow as tf
import utils.config as cfg
from utils.evaluate import plot_classification_report, plot_confusion_matrix
from utils.evaluate import f1


def predict(model, out_path):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TEST_DATA_DIR, image_size=cfg.IMAGE_SIZE,
        seed=42, color_mode="rgb")
    
    labels = test_ds.class_names
    labels_idx = [i for i in range(len(labels))]

    images, y_true = test_ds

    y_pred = model.predict(test_ds)
    
    f1_score = f1(y_true, y_pred)
    
    for i, score in enumerate(f1_score):
        print(f"Class {labels[i]} achieved an F1 score of {score}")
        
    plot_confusion_matrix(y_true, y_pred, labels_idx, labels, out_path)
    plot_classification_report(y_true, y_pred, labels_idx, labels, out_path)


if __name__ == "__main__":
    model = tf.keras.models.load_model(cfg.MODEL_PATH)
    path = os.getcwd()
    
    predict(model, path)
