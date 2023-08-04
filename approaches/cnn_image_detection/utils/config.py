##### GENERAL CONFIG #####
DEBUG = True
OBJECT_DETECTION = False


##### DATA CONFIG #####
N_WINDOWS = 100
DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
DRIFT_TYPES = ["sudden","gradual","incremental","recurring"]
DISTANCE_MEASURE = "cos"
COLOR = "color"
P_MODE = "train"


##### MODEL CONFIG #####
IMAGE_SIZE = (512, 512)
TARGETSIZE = 512
FC_LAYER = [128]
L_R = 3e-4
DROPOUT = 0.25
N_CLASSES = len(DRIFT_TYPES)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 64
EPOCHS = 200

# Boolean Parameters
PRETRAINED = False
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

# resnet, inception, inc_res, resnet_rs, xception, baseline
MODEL_SELECTION = "xception"
# average, flatten
AGG_LAYER = "average"
# adam, adadelta, adagrad, sgd, rms_p
OPTIMIZER = "adam"


##### IMAGE CLASSIFICATION CONFIG #####
TRAIN_DATA_DIR = ""
EVAL_DATA_DIR = ""
TEST_DATA_DIR = ""
MODEL_PATH = ""
DEFAULT_OUTPUT_DIR = "approaches\\cnn_image_detection\\output"
TENSORBOARD_DIR = "approaches\\cnn_image_detection\\tensorboard_logs"