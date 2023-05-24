##### GENERAL CONFIG #####
DEBUG = True
OBJECT_DETECTION = True


##### DATA CONFIG #####
N_WINDOWS = 100
DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
DRIFT_TYPES = ["sudden","gradual","incremental","recurring"]


##### MODEL CONFIG #####
IMAGE_SIZE = (256, 256)
TARGETSIZE = 256
N_CLASSES = len(DRIFT_TYPES)
BATCH_SIZE = 32
HEIGHT, WIDTH = 256, 256
TRAIN_STEPS = 30000
VAL_STEPS = 300
STEPS_PER_LOOP = 300
SUMMARY_INTERVAL = 300
CP_INTERVAL = 300
VAL_INTERVAL = 300
LR_INITIAL = 0.1
LR_WARMUP = 0.05
LR_WARMUP_STEPS = 300

BEST_CP_DIR = "approaches\\object_segmentation\\model_logging\\best_cp"
BEST_CP_METRIC = "AP"
BEST_CP_METRIC_COMP = "higher"

OPTIMIZER_TYPE = "adam"

# Possible Models:
# retinanet_resnetfpn_coco, retinanet_spinenet_coco
MODEL_SELECTION = "retinanet_spinenet_coco"

LR_DECAY = True

##### OBJECT DETECTION CONFIG #####
N_SHARDS = 1
TRAIN_DATA_DIR = ""
EVAL_DATA_DIR = ""
TEST_DATA_DIR = ""
MODEL_PATH = "approaches\\object_segmentation\\model_logging"
TFR_RECORDS_DIR = "approaches\\object_segmentation\\tfr_data"
DEFAULT_OUTPUT_DIR = "approaches\\object_segmentation\\output"
TRAINED_MODEL_PATH = ""

