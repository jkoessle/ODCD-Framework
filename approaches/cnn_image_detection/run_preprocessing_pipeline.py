import cnn_approach.preprocessing_pipeline as pp
import utils.config as cfg


if __name__ == "__main__":
    
    if cfg.MULTILABEL:
        pp.preprocessing_pipeline_multilabel(cfg.N_WINDOWS, p_mode=cfg.P_MODE)
    else:
        pp.preprocessing_pipeline(cfg.N_WINDOWS, p_mode=cfg.P_MODE)