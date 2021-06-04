import os
import shutil
import wget
import glob
import common_config

labels = common_config.model_config["labels"]

if not os.path.exists(common_config.ANNOTATION_PATH):
    os.mkdir(common_config.ANNOTATION_PATH)

with open(f'{common_config.ANNOTATION_PATH}/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

os.system(f'python {common_config.SCRIPTS_PATH}/generate_tfrecord.py -x {common_config.IMAGE_PATH}/train -l {common_config.ANNOTATION_PATH}/label_map.pbtxt -o {common_config.ANNOTATION_PATH}/train.record')
os.system(f'python {common_config.SCRIPTS_PATH}/generate_tfrecord.py -x {common_config.IMAGE_PATH}/test -l {common_config.ANNOTATION_PATH}/label_map.pbtxt -o {common_config.ANNOTATION_PATH}/test.record')

os.system(f'wget {common_config.model_config["pretrained_model_url"]} -P {common_config.PRETRAINED_MODEL_PATH}')

compressed_pretrained_model = glob.glob(f'{common_config.PRETRAINED_MODEL_PATH}/*.tar.gz')[0]
shutil.unpack_archive(compressed_pretrained_model, extract_dir=common_config.PRETRAINED_MODEL_PATH)
os.remove(compressed_pretrained_model)

pretrained_model_name = next(os.scandir(common_config.PRETRAINED_MODEL_PATH)).name

if os.path.exists(common_config.MODEL_PATH) and os.getenv('CLEAN_UP_WORKSPACE') == '1':
    shutil.rmtree(common_config.MODEL_PATH)

if not os.path.exists(common_config.MODELS_PATH):
    os.mkdir(common_config.MODELS_PATH)

if not os.path.exists(common_config.MODEL_PATH):
    os.mkdir(common_config.MODEL_PATH)

CONFIG_PATH = f'{common_config.PRETRAINED_MODEL_PATH}/{pretrained_model_name}/pipeline.config'

shutil.copy(CONFIG_PATH, common_config.MODEL_PATH)



import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(common_config.CUSTOM_MODEL_CONFIG, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = common_config.model_config['train_batch_size']
pipeline_config.train_config.fine_tune_checkpoint = f'{common_config.PRETRAINED_MODEL_PATH}/{pretrained_model_name}/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= f'{common_config.ANNOTATION_PATH}/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [f'{common_config.ANNOTATION_PATH}/train.record']
pipeline_config.eval_input_reader[0].label_map_path = f'{common_config.ANNOTATION_PATH}/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [f'{common_config.ANNOTATION_PATH}/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(common_config.CUSTOM_MODEL_CONFIG, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)

with open(f'/train.sh', 'w') as f:
    f.write(f'python {common_config.APIMODEL_PATH}/research/object_detection/model_main_tf2.py '
            f'--model_dir={common_config.MODEL_PATH} '
            f'--pipeline_config_path={common_config.CUSTOM_MODEL_CONFIG} '
            f'--num_train_steps={common_config.model_config["num_train_steps"]}')

with open(f'/export_tflite.sh', 'w') as f:
    f.write(f'python {common_config.APIMODEL_PATH}/research/object_detection/export_tflite_graph_tf2.py '
            f'--pipeline_config_path={common_config.CUSTOM_MODEL_CONFIG} '
            f'--trained_checkpoint_dir={common_config.CHECKPOINT_PATH} '
            f'--output_directory={common_config.MODEL_PATH}')