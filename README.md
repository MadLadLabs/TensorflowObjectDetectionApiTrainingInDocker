# TensorflowObjectDetectionApiTrainingInDocker




Build and run container for testing via
```
docker build -t tfod_test .
docker run --volume path/to/workspace:/tf-workspace --gpus all -it tfod_test bash
```

Object detection tested within the container using
```
cd ~/tf/models/research/
python object_detection/builders/model_builder_tf2_test.py
```