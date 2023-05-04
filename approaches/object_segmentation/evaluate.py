import tensorflow as tf

import utils.config as cfg
import utils.utilities as utils


def evaluate(model):
    utils.visualize_batch(path=cfg.EVAL_DATA_DIR, mode="validation")
    utils.visualize_predictions(path=cfg.EVAL_DATA_DIR,
                                mode="validation",
                                model=model)


if __name__ == "__main__":

    model = tf.saved_model.load(cfg.DEFAULT_OUTPUT_DIR)
    
    evaluate(model)
