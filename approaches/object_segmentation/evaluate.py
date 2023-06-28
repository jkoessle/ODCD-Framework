# import logging, os

# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import tensorflow as tf

import utils.config as cfg
import utils.utilities as utils
import utils.evaluation as eval


def visualize_bboxes(model):
    seed = random.randint(0, 10000)

    utils.visualize_batch(path=cfg.EVAL_DATA_DIR,
                          mode="validation",
                          seed=seed)
    utils.visualize_predictions(path=cfg.EVAL_DATA_DIR,
                                mode="validation",
                                model=model,
                                seed=seed,
                                threshold=cfg.EVAL_THRESHOLD)


if __name__ == "__main__":
    # tf.get_logger().setLevel('ERROR')

    # from absl import logging

    # tf.autograph.set_verbosity(0)
    model = tf.saved_model.load(cfg.TRAINED_MODEL_PATH)
    visualize_bboxes(model)

    eval.evaluate(data_dir=cfg.TEST_IMAGE_DATA_DIR,
                  eval_dir=cfg.EVAL_DATA_DIR,
                  model=model,
                  threshold=cfg.EVAL_THRESHOLD)
