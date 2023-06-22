# import logging, os

# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import tensorflow as tf

import utils.config as cfg
import utils.utilities as utils


def visualize_bboxes(model):
    seed = random.randint(0, 10000)
    
    utils.visualize_batch(path=cfg.EVAL_DATA_DIR, 
                          mode="validation", 
                          seed=seed)
    utils.visualize_predictions(path=cfg.EVAL_DATA_DIR,
                                mode="validation",
                                model=model,
                                seed=seed)
    

if __name__ == "__main__":
    # tf.get_logger().setLevel('ERROR')
    
    # from absl import logging

    # tf.autograph.set_verbosity(0)
    model = tf.saved_model.load(cfg.TRAINED_MODEL_PATH)
    threshold = 0.5
    
    visualize_bboxes(model)
