##### GENERAL CONFIG #####
DEBUG = True
OBJECT_DETECTION = True


##### DATA CONFIG #####
N_WINDOWS = 100
DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
DRIFT_TYPES = ["sudden","gradual","incremental","recurring"]


##### MODEL CONFIG #####
IMAGE_SIZE = (150, 150)
TARGETSIZE = 150
FC_LAYER = [128]
L_R = 3e-4
DROPOUT = 0.25
N_CLASSES = len(DRIFT_TYPES)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 64
EPOCHS = 200


##### OBJECT DETECTION CONFIG #####
N_SHARDS = 1
TRAIN_DATA_DIR = ""
EVAL_DATA_DIR = ""
TEST_DATA_DIR = ""
MODEL_PATH = "approaches\\object_segmentation\\model_logging"
TFR_RECORDS_DIR = "approaches\\object_segmentation\\tfr_data"
DEFAULT_OUTPUT_DIR = "approaches\\object_segmentation\\output"