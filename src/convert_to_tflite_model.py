import tensorflow as tf
import common_config



# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(common_config.TFLITE_GRAPH_PATH) # path to the SavedModel directory

################ https://www.tensorflow.org/lite/performance/post_training_quantization
################ Add this for optimization

# def representative_dataset():
#   for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
#     yield [tf.dtypes.cast(data, tf.float32)]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset

################ Add this for integer only operation

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8  # or tf.int8
# converter.inference_output_type = tf.uint8  # or tf.int8


tflite_model = converter.convert()

# Save the model.
with open('f{common_config.TFLITE_MODEL_FILE_PATH}', 'wb') as f:
  f.write(tflite_model)