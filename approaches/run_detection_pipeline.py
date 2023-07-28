import os
import pandas as pd

import object_segmentation.preprocessing as pp
import object_segmentation.utils.config as cfg
import object_segmentation.utils.utilities as seg_utils
import object_segmentation.utils.vdd_helper as vdd

def generate_only_annotations(data_dir: str):
    """Generate only annotations for given data

    Args:
        data_dir (str): Image data directory
    """
    drift_info = pd.read_csv(os.path.join(data_dir, "drift_info.csv"))
    log_matching = pd.read_csv(os.path.join(data_dir, "log_matching.csv"))
    
    log_matching = dict(log_matching.values)
    
    log_names = log_matching.keys()
    
    if cfg.VDD_PREPROCESSING:
        print("Generating VDD annotations")
        vdd.generate_vdd_annotations(drift_info=drift_info,
                                     dir=data_dir,
                                     log_matching=log_matching,
                                     log_names=log_names)
    else:
        print("Generating WINSIM annotations")
        seg_utils.generate_annotations(drift_info=drift_info,
                                dir=data_dir,
                                log_matching=log_matching,
                                log_names=log_names)
    if cfg.AUTOMATE_TFR_SCRIPT:
        seg_utils.start_tfr_script(repo_dir=cfg.TENSORFLOW_MODELS_DIR,
                                data_dir=cfg.DEFAULT_DATA_DIR,
                                tfr_dir=cfg.TFR_RECORDS_DIR,
                                prefix=cfg.OUTPUT_PREFIX)


if __name__ == "__main__":
    if cfg.ANNOTATIONS_ONLY:
        generate_only_annotations(cfg.DEFAULT_DATA_DIR)
    elif cfg.VDD_PREPROCESSING:
        print("Starting VDD pipeline")
        pp.vdd_pipeline()
    else:
        print("Starting WINSIM pipeline")
        pp.winsim_pipeline(n_windows=cfg.N_WINDOWS, p_mode=cfg.P_MODE)
    
    