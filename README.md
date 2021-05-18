# Tensorflow Object Detection Api Training In Docker

The purpose of these images is to allow you to quickly train an object detection model by transfer learning onto a model from the Tensorflow Object Detection Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Big thanks to people who publish models like these.

## How to use these images

It's assumed that you are starting with unlabeled images and will want to use labelImage by tzutalin.

1. Clone the repo to get the workspace scaffolded (git clone https://github.com/MadLadLabs/TensorflowObjectDetectionApiTrainingInDocker.git) and cd into it

2. Put your unlabeled images in the `workspace/images/raw` directory

3. Run labelImg (if you have it locally or use the snipped below to use it from a container) and label your images.

~~~
xhost +

docker run -ti --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/workspace/images/raw:/images/ \
    spiridonovpolytechnic/labelimg:latest
~~~


4. Move your images from the `workspace/images/raw` directory to the `workspace/images/train` and `workspace/images/test` directories. Typically, you might have 85% training data and 15% test data or thereabouts.

5. Update the `workspace/config.yml` file. The labels array should match the names of the objects you have labeled in the images. You can swap out the pretrained model for another from the object detection model zoo.

6. Run the image with the one of the training tags (latest or latest-gpu if you would like to use GPU to train)

~~~
docker run \
    --volume $(pwd):/tf-workspace \
    --gpus all -it \
    spiridonovpolytechnic/tensorflow-object-detection-training:latest-gpu
~~~

7. Once the training finishes, you can run another image to test inferrence using your webcam with one of the images tagged with latest-infer-webcam or latest-infer-webcam-gpu. ALternative, you can use latest-infer-file-to-screen or latest-infer-file-to-screen-gpu tags to test inferrence against an input video file.

~~~
docker run -e DISPLAY=$DISPLAY \
    --device=/dev/video0:/dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --volume $(pwd):/tf-workspace \
    --gpus all \
    spiridonovpolytechnic/tensorflow-object-detection-training:latest-infer-webcam-gpu
~~~

~~~
docker run -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --volume $(pwd):/tf-workspace \
    --volume /path/to/directory_with_input_video:/input/ \
    -e "INPUT=name_of_input_video_file" \
    --gpus all \
    spiridonovpolytechnic/tensorflow-object-detection-training:latest-infer-file-to-screen-gpu
~~~

Build and run container locally for testing via
~~~
docker build -t tfod_test -f Dockerfile.gpu .
docker run --volume $(pwd):/tf-workspace --gpus all -it tfod_test bash
~~~

Object detection tested within the container using
~~~
cd ~/tf/models/research/
python object_detection/builders/model_builder_tf2_test.py
~~~


## Credit

The containers are originally based on code from a tutorial project by nicknochnack (https://github.com/nicknochnack/RealTimeObjectDetection).
I hit a couple of snags when trying to set up my environment, got frustrated while trying to get all the right dependencies and decided to containerize the process.