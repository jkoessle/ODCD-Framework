##### GENERAL CONFIG #####
DEBUG = True
OBJECT_DETECTION = True
ANNOTATIONS_ONLY = False
AUTOMATE_TFR_SCRIPT = True
VDD_PREPROCESSING = True


##### DATA CONFIG #####
N_WINDOWS = 200
DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
TENSORFLOW_MODELS_DIR = ""
OUTPUT_PREFIX = ""
DRIFT_TYPES = ["sudden", "gradual", "incremental", "recurring"]
DISTANCE_MEASURE = "cos"
COLOR = "color"
P_MODE = "train"
RESIZE_SUDDEN_BBOX = True
RESIZE_VALUE = 5


##### MODEL CONFIG #####
FACTOR = 500
TRAIN_EXAMPLES = 1000
EVAL_EXAMPLES = 500
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 4
STEPS_PER_LOOP = TRAIN_EXAMPLES // TRAIN_BATCH_SIZE
TRAIN_STEPS = FACTOR * STEPS_PER_LOOP
VAL_STEPS = EVAL_EXAMPLES // EVAL_BATCH_SIZE
SUMMARY_INTERVAL = STEPS_PER_LOOP
CP_INTERVAL = STEPS_PER_LOOP
VAL_INTERVAL = STEPS_PER_LOOP

IMAGE_SIZE = (256, 256)
TARGETSIZE = 256
N_CLASSES = len(DRIFT_TYPES)
SCALE_MAX = 2.0
SCALE_MIN = 0.1

HEIGHT, WIDTH = 256, 256
LR_DECAY = True
LR_INITIAL = 1e-3
LR_WARMUP = 2.5e-4
LR_WARMUP_STEPS = 0.1 * TRAIN_STEPS

BEST_CP_METRIC = "AP"
BEST_CP_METRIC_COMP = "higher"

OPTIMIZER_TYPE = "adam"
LR_TYPE = "cosine"

SGD_MOMENTUM = 0.9
SGD_CLIPNORM = 10.0

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999

STEPWISE_BOUNDARIES = [0.95 * TRAIN_STEPS,
                       0.98 * TRAIN_STEPS]
STEPWISE_VALUES = [0.32 * TRAIN_BATCH_SIZE / 256.0,
                   0.032 * TRAIN_BATCH_SIZE / 256.0,
                   0.0032 * TRAIN_BATCH_SIZE / 256.0]


# Possible Models:
# retinanet_resnetfpn_coco, retinanet_spinenet_coco
MODEL_SELECTION = "retinanet_spinenet_coco"
SPINENET_ID = "143"


##### OBJECT DETECTION CONFIG #####
TRAIN_DATA_DIR = ""
EVAL_DATA_DIR = ""
TEST_DATA_DIR = ""
MODEL_PATH = ""
TFR_RECORDS_DIR = ""
DEFAULT_OUTPUT_DIR = ""
TRAINED_MODEL_PATH = ""


##### VDD CONFIG #####
SUB_L = 200
SLI_BY = 75
CP_ALL = True