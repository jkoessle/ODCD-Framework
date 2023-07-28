import tensorflow_models as tfm
import tensorflow as tf
import os
import argparse

import utils.config as cfg
import utils.utilities as utils

from official.vision.serving import export_saved_model_lib
from typing import Tuple


def train(model_dir=cfg.MODEL_PATH, 
          output_dir=cfg.DEFAULT_OUTPUT_DIR) -> Tuple[tf.keras.Model, dict]:
    """Main function for training the object detection models.

    Args:
        model_dir (str, optional): TensorFlow Model path. Defaults to cfg.MODEL_PATH.
        output_dir (str, optional): Output directory. 
            Defaults to cfg.DEFAULT_OUTPUT_DIR.

    Returns:
        Tuple[tf.keras.Model, dict]: TensorFlow model and evaluation logs
    """

    timestamp = utils.get_timestamp()

    model_dir = os.path.join(model_dir,
                             f"{timestamp}_{cfg.ENCODING_TYPE}_{cfg.OPTIMIZER_TYPE}")
    output_dir = os.path.join(output_dir,
                              f"{timestamp}_{cfg.ENCODING_TYPE}_{cfg.OPTIMIZER_TYPE}")

    exp_config = utils.get_model_config(model_dir)

    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    logical_device_names = [
        logical_device.name for logical_device in tf.config.list_logical_devices()]

    if 'GPU' in ''.join(logical_device_names):
        distribution_strategy = tf.distribute.MirroredStrategy()
    else:
        print('Warning: this will be really slow. Using CPU')
        distribution_strategy = tf.distribute.OneDeviceStrategy(
            logical_device_names[0])

    print('Setup Done')

    with distribution_strategy.scope():
        task = tfm.core.task_factory.get_task(
            exp_config.task, logging_dir=model_dir)

    for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
        print()
        print(
            f'images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}')
        print(f'labels.keys: {labels.keys()}')

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True)

    save_options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)

    export_saved_model_lib.export_inference_graph(
        input_type='image_tensor',
        batch_size=1,
        input_image_size=[cfg.HEIGHT, cfg.WIDTH],
        params=exp_config,
        log_model_flops_and_params=True,
        save_options=save_options,
        checkpoint_path=tf.train.latest_checkpoint(model_dir),
        export_dir=output_dir)

    return model, eval_logs


if __name__ == "__main__":

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--gpu_devices", dest="gpu_devices",
                        help="Specify which CUDA devices to use.",
                        default="",
                        type=str)
    args = parser.parse_args()

    # set cuda devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    # TODO handle model and eval_logs
    model, eval_logs = train()
