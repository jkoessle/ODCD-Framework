import cnn_approach.preprocessing_pipeline as pp
import utils.config as cfg


if __name__ == "__main__":
    
    if cfg.MULTILABEL:
        pp.preprocessing_pipeline_multilabel(100, p_mode="train")
    else:
        pp.preprocessing_pipeline(100, p_mode="train")