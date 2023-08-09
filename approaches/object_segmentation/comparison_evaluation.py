import os
import utils.evaluation as eval
import utils.utilities as utils
import utils.config as cfg

from tqdm import tqdm


def start_pro_drift_evaluation():
    """Handle for ProDrift evaluation.
    """
    files = utils.get_event_log_paths(cfg.DEFAULT_LOG_DIR)
    
    timestamp = utils.get_timestamp()
    
    out_path = os.path.abspath(os.path.join("evaluation_results",
                                            "ProDrift",
                                            f"{timestamp}_prodrift_results.txt"))
    
    for name, path in tqdm(files.items(), desc="Generate Predictions with ProDrift",
                           unit="Event Log"):
        log_path = os.path.join(path, name)
        outs, errs = eval.call_pro_drift(log_path, cfg.PRODRIFT_DIR, 100)
        with open(out_path, "a") as f:
            f.write(outs.decode())
            f.write("\n")
            
            
def start_vdd_evaluation():
    """Handle for VDD evaluation.
    """
    eval.call_vdd(log_dir=cfg.DEFAULT_LOG_DIR, 
                  vdd_dir=cfg.VDD_DIR)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Handles to start evaluation per tool
    # start_pro_drift_evaluation()
    # start_vdd_evaluation()
    
    # Transformations for results from ProDrift
    xslx_path = "evaluation_results\ProDrift\prodrift_results_manual.xlsx"
    csv_path = "evaluation_results\ProDrift\drift_info.csv"
    # eval.excel_2_csv(xslx_path, csv_path)
    # eval.preprocess_pro_drift_results(csv_path)
    
    eval.evaluate_pro_drift_results(results_file_path=csv_path, 
                                    data_dir=cfg.TEST_IMAGE_DATA_DIR)
    