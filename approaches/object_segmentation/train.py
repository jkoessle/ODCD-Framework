import tensorflow_models as tfm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import config as cfg

from official.core import exp_factory
from official.vision.serving import export_saved_model_lib
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder


def train(train_data_path=cfg.TRAIN_DATA_DIR, validation_data_path=cfg.EVAL_DATA_DIR,
          model_dir=cfg.MODEL_PATH, output_dir=cfg.DEFAULT_OUTPUT_DIR):

    exp_config = exp_factory.get_exp_config('retinanet_resnetfpn_coco')

    # non adjustable for pretrained model
    HEIGHT, WIDTH = 256, 256
    IMG_SIZE = [HEIGHT, WIDTH, 3]

    # Backbone config
    exp_config.task.freeze_backbone = False
    exp_config.task.annotation_file = ''

    # Model config
    exp_config.task.model.input_size = IMG_SIZE
    exp_config.task.model.num_classes = cfg.N_CLASSES + 1
    exp_config.task.model.detection_generator. \
        tflite_post_processing.max_classes_per_detection = \
        exp_config.task.model.num_classes

    # Training data config
    exp_config.task.train_data.input_path = train_data_path
    exp_config.task.train_data.dtype = 'float32'
    exp_config.task.train_data.global_batch_size = cfg.BATCH_SIZE
    exp_config.task.train_data.parser.aug_scale_max = 1.0
    exp_config.task.train_data.parser.aug_scale_min = 1.0

    # Validation data config.
    exp_config.task.validation_data.input_path = validation_data_path
    exp_config.task.validation_data.dtype = 'float32'
    exp_config.task.validation_data.global_batch_size = cfg.BATCH_SIZE

    logical_device_names = [
        logical_device.name for logical_device in tf.config.list_logical_devices()]

    # if 'GPU' in ''.join(logical_device_names):
    #     print('This may be broken in Colab.')
    #     device = 'GPU'
    # elif 'TPU' in ''.join(logical_device_names):
    #     print('This may be broken in Colab.')
    #     device = 'TPU'
    # else:
    #     print('Running on CPU is slow, so only train for a few steps.')
    #     device = 'CPU'

    train_steps = 1000
    # steps_per_loop = num_of_training_examples // train_batch_size
    exp_config.trainer.steps_per_loop = 100

    exp_config.trainer.summary_interval = 100
    exp_config.trainer.checkpoint_interval = 100
    exp_config.trainer.validation_interval = 100
    # validation_steps = num_of_validation_examples // eval_batch_size
    exp_config.trainer.validation_steps = 100
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
    exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    if 'GPU' in ''.join(logical_device_names):
        distribution_strategy = tf.distribute.MirroredStrategy()
    elif 'TPU' in ''.join(logical_device_names):
        tf.tpu.experimental.initialize_tpu_system()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='/device:TPU_SYSTEM:0')
        distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        print('Warning: this will be really slow.')
        distribution_strategy = tf.distribute.OneDeviceStrategy(
            logical_device_names[0])

    print('Done')

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
    
    print(eval_logs)

    export_saved_model_lib.export_inference_graph(
        input_type='image_tensor',
        batch_size=1,
        input_image_size=[HEIGHT, WIDTH],
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(model_dir),
        export_dir=output_dir)

    return exp_config


def get_ex_decoder():
    category_index = {
        1: {
            'id': 0,
            'name': 'sudden'
        },
        2: {
            'id': 1,
            'name': 'gradual'
        },
        3: {
            'id': 2,
            'name': 'incremental'
        },
        4: {
            'id': 3,
            'name': 'recurring'
        }
    }
    tf_ex_decoder = TfExampleDecoder()

    return category_index, tf_ex_decoder


def show_batch(exp_config, n_examples=4):

    category_index, tf_ex_decoder = get_ex_decoder()

    raw_data = tf.data.TFRecordDataset(
        exp_config.task.train_data.input_path).shuffle(
        buffer_size=20).take(n_examples)

    plt.figure(figsize=(20, 20))
    use_normalized_coordinates = True
    min_score_thresh = 0.30
    for i, serialized_example in enumerate(raw_data):
        plt.subplot(1, 3, i + 1)
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = decoded_tensors['image'].numpy().astype('uint8')
        scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            decoded_tensors['groundtruth_boxes'].numpy(),
            decoded_tensors['groundtruth_classes'].numpy().astype('int'),
            scores,
            category_index=category_index,
            use_normalized_coordinates=use_normalized_coordinates,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=4)

        # plt.imshow(image)
        plt.axis('off')
        plt.title(f'Image-{i+1}')
    plt.imsave(os.path.join(cfg.DEFAULT_OUTPUT_DIR, "batch.png"))


if __name__ == "__main__":

    exp_config = train()

    show_batch(exp_config=exp_config)
