##### GENERAL CONFIG #####
DEBUG = True
OBJECT_DETECTION = True
ANNOTATIONS_ONLY = False
AUTOMATE_TFR_SCRIPT = True
VDD_PREPROCESSING = False
KEEP_AXIS = False
WINDOWS_SYSTEM = True
MINE_CONSTRAINTS = True
CONSTRAINTS_DIR = ""

if VDD_PREPROCESSING:
    ENCODING_TYPE = "vdd"
else:
    ENCODING_TYPE = "winsim"


##### DATA CONFIG #####
N_WINDOWS = 200
DEFAULT_DATA_DIR = "Specify default data output directory"
DEFAULT_LOG_DIR = "Specify event log directory"
TFR_RECORDS_DIR = "Specify directory where to save TFR files here"
TENSORFLOW_MODELS_DIR = "Specify TensorFlow model garden directory"
MINERFUL_SCRIPTS_DIR = "Specify MINERful directory"
OUTPUT_PREFIX = "Specify output prefix for TFR file"
DRIFT_TYPES = ["sudden", "gradual", "incremental", "recurring"]
DISTANCE_MEASURE = "cos" # can be one of ["fro","nuc","inf","l2","cos","earth"]
COLOR = "color"
RESIZE_SUDDEN_BBOX = True
RESIZE_VALUE = 5

##### VDD CONFIG #####
SUB_L = 100
SLI_BY = 50
CP_ALL = True


##### MODEL CONFIG #####
FACTOR = 500
TRAIN_EXAMPLES = 9978
EVAL_EXAMPLES = 2495
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 32
STEPS_PER_LOOP = TRAIN_EXAMPLES // TRAIN_BATCH_SIZE
TRAIN_STEPS = FACTOR * STEPS_PER_LOOP
VAL_STEPS = EVAL_EXAMPLES // EVAL_BATCH_SIZE
SUMMARY_INTERVAL = STEPS_PER_LOOP
CP_INTERVAL = STEPS_PER_LOOP
VAL_INTERVAL = STEPS_PER_LOOP
EVAL_THRESHOLD = 0.5

# must be equally sized!
IMAGE_SIZE = (256, 256)
TARGETSIZE = 256
N_CLASSES = len(DRIFT_TYPES)
SCALE_MAX = 2.0
SCALE_MIN = 0.1

WIDTH, HEIGHT  = IMAGE_SIZE 
LR_DECAY = True
LR_INITIAL = 1e-3
LR_WARMUP = 2.5e-4
LR_WARMUP_STEPS = 0.1 * TRAIN_STEPS

BEST_CP_METRIC = "AP"
BEST_CP_METRIC_COMP = "higher"

OPTIMIZER_TYPE = "sgd"
LR_TYPE = "stepwise"

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
# ID can be 143 or 190
SPINENET_ID = "143"


##### OBJECT DETECTION CONFIG #####
TRAIN_DATA_DIR = "Specify path to TFR training dataset here"
EVAL_DATA_DIR = "Specify path to TFR validation dataset here"
MODEL_PATH = "Specify directory where to log model training here"
DEFAULT_OUTPUT_DIR = "Specify directory where to save output here"
TRAINED_MODEL_PATH = "Specify path to trained model here"

TEST_IMAGE_DATA_DIR = "Specify directory where evaluation images are saved here"


##### EVALUATION CONFIG #####
RELATIVE_LAG = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
EVAL_MODE = "general"
PRODRIFT_DIR = "Specify directory where ProDrift is stored"
VDD_DIR = "Specify directory where VDD is stored"