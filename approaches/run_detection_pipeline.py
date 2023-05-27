import object_segmentation.preprocessing as pp
import object_segmentation.utils.config as cfg


if __name__ == "__main__":
    pp.preprocessing_pipeline_multilabel(n_windows=cfg.N_WINDOWS, p_mode=cfg.P_MODE)