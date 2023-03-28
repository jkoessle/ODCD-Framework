DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
INTERIM_DATA_DIR = "data\\train"
DEFAULT_OUTPUT_DIR = ""
N_WINDOWS = 100
MODE = "time"

# resnet, inception, inc_res, resnet_rs, xception
MODEL_SELECTION = "resnet"

TARGETSIZE = 150
FC_LAYER = [1024, 512, 256]
L_R = 3e-4
DROPOUT = 0.25
N_CLASSES = 3
PRETRAINED = False

# average, flatten
AGG_LAYER = "flatten"

# adam, adadelta, adagrad, sgd, rms_p
OPTIMIZER = "adam"

EPOCHS = 200

DRIFT_TYPES = ["no_drift","gradual","sudden","incremental","recurring"]

PREPROCESS = False
CHECKPOINTS = False
EARLY_STOPPING = True
TENSORBOARD = True
SAVE_MODEL = True
AUGMENTATION = True