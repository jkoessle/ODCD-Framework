import os
import pandas as pd

import object_segmentation.preprocessing as pp
import object_segmentation.utils.config as cfg
import object_segmentation.utils.utilities as seg_utils


def generate_only_annotations(data_dir):
    drift_info = pd.read_csv(os.path.join(data_dir, "drift_info.csv"))
    log_matching = pd.read_csv(os.path.join(data_dir, "log_matching.csv"))
    
    log_matching = dict(log_matching.values)
    
    log_names = log_matching.keys()
    
    seg_utils.generate_annotations(drift_info=drift_info,
                               dir=data_dir,
                               log_matching=log_matching,
                               log_names=log_names)


if __name__ == "__main__":
    if cfg.ANNOTATIONS_ONLY:
        generate_only_annotations(cfg.DEFAULT_DATA_DIR)
    elif cfg.VDD_PREPROCESSING:
        pp.vdd_pipeline()
    else:
        pp.preprocessing_pipeline_multilabel(n_windows=cfg.N_WINDOWS, p_mode=cfg.P_MODE)
    
    