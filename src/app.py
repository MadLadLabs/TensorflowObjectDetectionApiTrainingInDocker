import os
import shutil
import yaml
import wget
import glob


WORKSPACE_PATH = '/tf-workspace'
SCRIPTS_PATH = '/tf/scripts'
APIMODEL_PATH = '/tf/models'
PRETRAINED_MODEL_PATH = f'/tf/pre-trained-models'
ANNOTATION_PATH = f'{WORKSPACE_PATH}/annotations'
IMAGE_PATH = f'{WORKSPACE_PATH}/images'
MODELS_PATH = f'{WORKSPACE_PATH}/models'

with open(f'{WORKSPACE_PATH}/config.yml') as model_config_file:
    model_config = yaml.load(model_config_file, Loader=yaml.FullLoader)

MODEL_NAME = model_config["name"]
MODEL_PATH = f'{MODELS_PATH}/{MODEL_NAME}'

CONFIG_PATH = f'{MODEL_PATH}/pipeline.config'
CHECKPOINT_PATH = f'{MODEL_PATH}/'

labels = model_config["labels"]

if not os.path.exists(ANNOTATION_PATH):
    os.mkdir(ANNOTATION_PATH)

with open(f'{ANNOTATION_PATH}/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

os.system(f'python {SCRIPTS_PATH}/generate_tfrecord.py -x {IMAGE_PATH}/train -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/train.record')
os.system(f'python {SCRIPTS_PATH}/generate_tfrecord.py -x {IMAGE_PATH}/test -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/test.record')

os.system(f'wget {model_config["pretrained_model_url"]} -P {PRETRAINED_MODEL_PATH}')

compressed_pretrained_model = glob.glob(f'{PRETRAINED_MODEL_PATH}/*.tar.gz')[0]
shutil.unpack_archive(compressed_pretrained_model, extract_dir=PRETRAINED_MODEL_PATH)
os.remove(compressed_pretrained_model)

pretrained_model_name = next(os.scandir(PRETRAINED_MODEL_PATH)).name

if os.path.exists(MODEL_PATH) and os.getenv('CLEAN_UP_WORKSPACE') == '1':
    shutil.rmtree(MODEL_PATH)

if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

CONFIG_PATH = f'{PRETRAINED_MODEL_PATH}/{pretrained_model_name}/pipeline.config'

shutil.copy(CONFIG_PATH, MODEL_PATH)

CUSTOM_MODEL_CONFIG = f'{MODEL_PATH}/pipeline.config'

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CUSTOM_MODEL_CONFIG, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = model_config['train_batch_size']
pipeline_config.train_config.fine_tune_checkpoint = f'{PRETRAINED_MODEL_PATH}/{pretrained_model_name}/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= f'{ANNOTATION_PATH}/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [f'{ANNOTATION_PATH}/train.record']
pipeline_config.eval_input_reader[0].label_map_path = f'{ANNOTATION_PATH}/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [f'{ANNOTATION_PATH}/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CUSTOM_MODEL_CONFIG, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)

with open(f'/train.sh', 'w') as f:
    f.write(f'python {APIMODEL_PATH}/research/object_detection/model_main_tf2.py '
            f'--model_dir={MODEL_PATH} '
            f'--pipeline_config_path={CUSTOM_MODEL_CONFIG} '
            f'--num_train_steps={model_config["num_train_steps"]}')