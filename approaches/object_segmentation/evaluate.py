import tensorflow as tf

from . import config as cfg
from . import utilities as utils


def evaluate(model):
    utils.visualize_batch(path=cfg.DEFAULT_OUTPUT_DIR, mode="validation")
    utils.visualize_predictions(path=cfg.DEFAULT_OUTPUT_DIR,
                                mode="validation",
                                model=model)


if __name__ == "__main__":

    model = tf.saved_model.load(cfg.DEFAULT_OUTPUT_DIR)

    utils.visualize_batch(cfg.DEFAULT_OUTPUT_DIR, mode="validation")
    utils.visualize_predictions(path=cfg.DEFAULT_OUTPUT_DIR,
                                mode="validation",
                                model=model)
