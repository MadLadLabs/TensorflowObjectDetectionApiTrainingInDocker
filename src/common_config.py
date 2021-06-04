import yaml

WORKSPACE_PATH = '/tf-workspace'
SCRIPTS_PATH = '/tf/scripts'
APIMODEL_PATH = '/tf/models'
PRETRAINED_MODEL_PATH = f'/tf/pre-trained-models'
ANNOTATION_PATH = f'{WORKSPACE_PATH}/annotations'
IMAGE_PATH = f'{WORKSPACE_PATH}/images'
MODELS_PATH = f'{WORKSPACE_PATH}/models'

model_config = None

with open(f'{WORKSPACE_PATH}/config.yml') as model_config_file:
    model_config = yaml.load(model_config_file, Loader=yaml.FullLoader)

MODEL_NAME = model_config["name"]
MODEL_PATH = f'{MODELS_PATH}/{MODEL_NAME}'
TFLITE_GRAPH_PATH = f'{MODELS_PATH}/{MODEL_NAME}-tflite-graph'
TFLITE_MODEL_PATH = f'{MODELS_PATH}/{MODEL_NAME}-tflite-model'
TFLITE_MODEL_FILE_PATH = f'{TFLITE_MODEL_PATH}/{MODEL_NAME}.tflite'

CONFIG_PATH = f'{MODEL_PATH}/pipeline.config'
CHECKPOINT_PATH = f'{MODEL_PATH}/'

CUSTOM_MODEL_CONFIG = f'{MODEL_PATH}/pipeline.config'