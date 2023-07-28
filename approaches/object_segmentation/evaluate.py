import random
import tensorflow as tf

import utils.config as cfg
import utils.utilities as utils
import utils.evaluation as eval


def visualize_bboxes(model: tf.keras.Model):
    """Call visualizing functions.

    Args:
        model (tf.keras.Model): TensorFlow model
    """
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
    model = tf.saved_model.load(cfg.TRAINED_MODEL_PATH)
    visualize_bboxes(model)

    eval.evaluate(data_dir=cfg.TEST_IMAGE_DATA_DIR,
                  model=model,
                  threshold=cfg.EVAL_THRESHOLD)
