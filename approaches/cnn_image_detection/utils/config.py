DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = ""

TRAIN_DATA_DIR = ""
EVAL_DATA_DIR = ""
TEST_DATA_DIR = ""
INTERIM_DATA_DIR = ""
TFR_RECORDS_DIR = "tfr_data"
DEFAULT_OUTPUT_DIR = "output"
MODEL_PATH = ""
N_WINDOWS = 100

OBJECT_DETECTION = True
DRIFT_TYPES = ["sudden","gradual","incremental","recurring"]

# resnet, inception, inc_res, resnet_rs, xception, baseline
MODEL_SELECTION = "baseline"

IMAGE_SIZE = (150, 150)
TARGETSIZE = 150
FC_LAYER = [128]
L_R = 3e-4
DROPOUT = 0.25
N_CLASSES = len(DRIFT_TYPES)
PRETRAINED = False
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 64

# average, flatten
AGG_LAYER = "average"

# adam, adadelta, adagrad, sgd, rms_p
OPTIMIZER = "adam"

EPOCHS = 200

PREPROCESS = False
CHECKPOINTS = True
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
TENSORBOARD = True
SAVE_MODEL = True
AUGMENTATION = False
XAI_VIS = True
NEW_MODEL = True
TRAIN_MODEL = True
PREDICT = True
MULTILABEL = True