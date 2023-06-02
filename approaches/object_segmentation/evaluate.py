# import logging, os

# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import utils.config as cfg
import utils.utilities as utils


def evaluate(model):
    utils.visualize_batch(path=cfg.EVAL_DATA_DIR, mode="validation")
    utils.visualize_predictions(path=cfg.EVAL_DATA_DIR,
                                mode="validation",
                                model=model)
    

if __name__ == "__main__":
    # tf.get_logger().setLevel('ERROR')
    
    # from absl import logging

    # tf.autograph.set_verbosity(0)
    model = tf.saved_model.load(cfg.TRAINED_MODEL_PATH)
    threshold = 0.5
    
    utils.get_drift_moments(None, eval_dir=cfg.EVAL_DATA_DIR, model=model)
    
    # evaluate(model)
