# ODCD Framework
This framework enables the detection and localization of concept drift in process mining from event logs using state-of-the-art deep learning models for object detection. The framework provides an end-to-end platform for preprocessing event logs, training new models, and evaluating their performance. Within the framework, already trained models are also available that can be directly applied to event logs. 

<!-- GETTING STARTED -->
## Getting Started

To get the framework up and running follow these steps.

### Prerequisites

For full functionality, this framework requires the following software:
* Python 3.9
* TensorFlow Model Garden ([clone here](https://github.com/tensorflow/models))
* MINERful ([clone here](https://github.com/cdc08x/MINERful)) -> Requires JRE 7+
* [poetry](https://python-poetry.org/) -> optional, for packaging/dependency management

<b>Note:</b> If you want to use only the pretrained models, the package dependencies in the poetry/requirements file are sufficient.

### Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/jkoessle/ODCD-Framework.git
   ```
2. Go into ODCD directory:
   ```sh
   cd ODCD-Framework
   ```
3. Install dependencies (one option is sufficient).
   
   With poetry:
   ```sh
   poetry install
   ```
   With pip: 
   ```sh
   pip install -r requirements.txt
   ```
   With conda environment:
   ```sh
   conda create --name odcd --file requirements.txt
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
The framework includes several use cases. These are explained below. 

### Pretrained Models
The main contribution of this framework are pretrained model, which can be directly used for detecting concept drift from event logs. 

All available pretrained models are listed in the table below. The corresponding training checkpoints can be found [here](https://data.dws.informatik.uni-mannheim.de/odcd-framework/models/object_detection/model_checkpoints/).
| **Model Name**   | **Pretrained Model**                                                                                           |
|------------------|----------------------------------------------------------------------------------------------------------------|
| vdd_fpn          | https://data.dws.informatik.uni-mannheim.de/odcd-framework/models/object_detection/models/vdd_fpn.zip          |
| vdd_spine_143    | https://data.dws.informatik.uni-mannheim.de/odcd-framework/models/object_detection/models/vdd_spine_143.zip    |
| vdd_spine_190    | https://data.dws.informatik.uni-mannheim.de/odcd-framework/models/object_detection/models/vdd_spine_190.zip    |
| winsim_fpn       | https://data.dws.informatik.uni-mannheim.de/odcd-framework/models/object_detection/models/winsim_fpn.zip       |
| winsim_spine_143 | https://data.dws.informatik.uni-mannheim.de/odcd-framework/models/object_detection/models/winsim_spine_143.zip |
| winsim_spine_190 | https://data.dws.informatik.uni-mannheim.de/odcd-framework/models/object_detection/models/winsim_spine_190.zip |


For this purpose, use the [predict](approaches\object_detection\predict.py) script. This script can either be used in an E2E method or applied directly to preprocessed images.
```sh
   cd approaches
   cd object_detection
```
Example for WINSIM with preprocessing:
```sh
   python predict.py --model-path <specify path of unzipped pretrained model> --log-dir <specify directory where event logs are stored> --encoding winsim --n-windows 200 --output-dir <specify output directory>
```
Example for WINSIM with preprocessed images:
```sh
   python predict.py --model-path <specify path of unzipped pretrained model> --image-dir <specify directory where preprocessed images are stored> --encoding winsim --n-windows 200 --output-dir <specify output directory>
```
Example for VDD with preprocessing:
```sh
   python predict.py --model-path <specify path of unzipped pretrained model> --log-dir <specify directory where event logs are stored> --encoding vdd --cp-all --output-dir <specify output directory>
```
Example for VDD with preprocessed images:
```sh
   python predict.py --model-path <specify path of unzipped pretrained model> --image-dir <specify directory where preprocessed images are stored> --encoding vdd --cp-all --output-dir <specify output directory>
```

The script outputs not only the visual detection of the drift types, but also a detailed report that specifies the drift moments on traces.
### E2E Platform
With the E2E platform it is possible to preprocess labeled event logs and train and validate new models with user defined parameters. For this purpose, you need to adjust the configuration to your needs.
#### Configuration Preprocessing
The configuration file can be found [here](approaches\object_detection\utils\config.py). For preprocessing the following variables must be set or adjusted:
```sh
##### GENERAL CONFIG #####
DEBUG = True
OBJECT_DETECTION = True
ANNOTATIONS_ONLY = False
AUTOMATE_TFR_SCRIPT = True
VDD_PREPROCESSING = True -> specifies which encoding type to use
KEEP_AXIS = True
WINDOWS_SYSTEM = True
MINE_CONSTRAINTS = False
CONSTRAINTS_DIR = ""

##### DATA CONFIG #####
N_WINDOWS = 200
DEFAULT_DATA_DIR = "Specify default data output directory"
DEFAULT_LOG_DIR = "Specify event log directory"
TENSORFLOW_MODELS_DIR = "Specify TensorFlow model garden directory" -> see    prerequisites
MINERFUL_SCRIPTS_DIR = "Specify MINERful directory" -> see prerequisites
OUTPUT_PREFIX = "Specify output prefix for TFR file"
DRIFT_TYPES = ["sudden", "gradual", "incremental", "recurring"]
DISTANCE_MEASURE = "cos" -> can be one of ["fro","nuc","inf","l2","cos","earth"]
COLOR = "color"
P_MODE = "train"
RESIZE_SUDDEN_BBOX = True
RESIZE_VALUE = 5

SUB_L = 100
SLI_BY = 50
CP_ALL = True
```

The preprocessing is then started with the [run_detection_pipeline](approaches\object_detection\run_detection_pipeline.py) script:
```sh
   cd approaches
   cd object_detection
```
```sh
   python run_detection_pipeline.py
```
If AUTOMATE_TFR_SCRIPT is True, this will output a TFR file that can be used for training
#### Configuration Training
For training the model, there are also important settings in [config](approaches\object_detection\utils\config.py) that must be adapted to your use:
```sh
##### MODEL CONFIG #####
FACTOR = 500
TRAIN_EXAMPLES = XXXX -> number of training examples (int)
EVAL_EXAMPLES = XXXX -> number of validation examples (int)
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 32

# Possible Models:
# retinanet_resnetfpn_coco, retinanet_spinenet_coco
MODEL_SELECTION = "Specify model selection"
# ID can be 143 or 190
SPINENET_ID = "143"

##### OBJECT DETECTION CONFIG #####
TRAIN_DATA_DIR = "Specify path to TFR training dataset here"
EVAL_DATA_DIR = "Specify path to TFR validation dataset here"
MODEL_PATH = "Specify directory where to log model training here"
DEFAULT_OUTPUT_DIR = "Specify directory where to save output here"
TRAINED_MODEL_PATH = "Specify path to trained model here"
```
The training is then started with the [train](approaches\object_detection\train.py) script:
```sh
   python train.py --gpu_devices 0,1
```
Here you can set the number of GPUs with the corresponding flag. This example would start the training on GPU 0 and GPU 1. The training produces tensorboard logs which are saved at the specified MODEL_PATH. The tensorboard can be started with:
```sh
   tensorboard --logdir <insert MODEL_PATH here> --bind_all --port <insert your port here> 
```

#### Configuration Evaluation
This framework also includes a separate evaluation of the trained models. For this purpose, the following parameter in the [config](approaches\object_detection\utils\config.py) file need to be set:
```sh
TEST_IMAGE_DATA_DIR = "Specify directory where evaluation images are saved here"

##### EVALUATION CONFIG #####
RELATIVE_LAG = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
EVAL_MODE = "general"
```
The evaluation is then started with the [evaluate](approaches\object_detection\evaluate.py) script:
```sh
   python evaluate.py
```
This script visualizes random entries from the specified test dataset and generates classification reports and performance reports for the entire data.

### Datasets
All TFR files, preprocessed images and event log datasets are available for download [here](https://data.dws.informatik.uni-mannheim.de/odcd-framework/data/).

The real event logs used during the evaluation can be found here:
- [BPIC 2011](https://doi.org/10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54)
- [BPIC 2012](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
- [Italian help desk](https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb)
- [Sepsis](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>