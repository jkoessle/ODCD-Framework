DEFAULT_DATA_DIR = "data\experiment_20230404-190632"
DEFAULT_LOG_DIR = "logs\\1680627785"

TRAIN_DATA_DIR = "data\experiment_20230404-190632"
EVAL_DATA_DIR = "data\experiment_20230404-190632"
TEST_DATA_DIR = "data\experiment_20230404-190632"
INTERIM_DATA_DIR = "data\experiment_20230404-190632"
DEFAULT_OUTPUT_DIR = "output"
MODEL_PATH = ""
N_WINDOWS = 100

DRIFT_TYPES = ["gradual","sudden"]

# resnet, inception, inc_res, resnet_rs, xception, baseline
MODEL_SELECTION = "baseline"

IMAGE_SIZE = (150, 150)
TARGETSIZE = 150
FC_LAYER = [128]
L_R = 3e-4
DROPOUT = 0.25
N_CLASSES = len(DRIFT_TYPES)
PRETRAINED = False

# average, flatten
AGG_LAYER = "flatten"

# adam, adadelta, adagrad, sgd, rms_p
OPTIMIZER = "adam"

EPOCHS = 200

PREPROCESS = False
CHECKPOINTS = True
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
TENSORBOARD = True
SAVE_MODEL = True
AUGMENTATION = True
XAI_VIS = True
NEW_MODEL = True
TRAIN_MODEL = True
PREDICT = True
MULTILABEL = True