import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


def r2l():
    return ReplaceToLinear()


def preprocess_model_input(model_selection, images):
    if model_selection == "resnet":
        p_images = tf.keras.applications.resnet_v2.preprocess_input(images)
    elif model_selection == "inception":
        p_images = tf.keras.applications.inception_v3.preprocess_input(images)
    elif model_selection == "inc_res":
        p_images = tf.keras.applications.inception_resnet_v2.preprocess_input(
            images)
    elif model_selection == "resnet_rs":
        p_images = tf.keras.applications.resnet_rs.preprocess_input(images)
    elif model_selection == "xception":
        p_images = tf.keras.applications.xception.preprocess_input(images)
    elif model_selection == "baseline":
        p_images = tf.keras.applications.resnet_v2.preprocess_input(images)
    return p_images


def smooth_grad(model, score, p_images, labels, path):

    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=r2l(),
                        clone=True)

    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(score,
                            p_images,
                            # The number of calculating gradients iterations.
                            smooth_samples=20,
                            smooth_noise=0.20)  # noise spread level.

    # Render
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
    for i, title in enumerate(labels):
        ax[i].set_title(title, fontsize=14)
        ax[i].imshow(saliency_map[i], cmap='jet')
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "images/smoothgrad.png"))


def grad_cam(model, score, images, p_images, labels, path):

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model,
                              model_modifier=r2l(),
                              clone=True)

    # Generate heatmap with GradCAM++
    cam = gradcam(score,
                  p_images,
                  penultimate_layer=-1)

    # Render
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
    for i, title in enumerate(labels):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "images/gradcam_plus_plus.png"))


def fast_score_cam(model, score, images, p_images, labels, path):

    # Create ScoreCAM
    scorecam = Scorecam(model, model_modifier=r2l())

    # Generate heatmap with Faster-ScoreCAM
    cam = scorecam(score,
                   p_images,
                   penultimate_layer=-1,
                   max_N=10)

    # Render
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
    for i, title in enumerate(labels):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "images/scorecam.png"))