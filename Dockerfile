FROM tensorflow/tensorflow:latest

ENV CLEAN_UP_WORKSPACE=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    protobuf-compiler \
    wget \
    git

RUN pip install \
    pyyaml \
    pandas \
    wget

RUN mkdir /tf
RUN mkdir /tf/scripts
RUN mkdir /tf/models

RUN wget \
    https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py \
    -P /tf/scripts

RUN git clone https://github.com/tensorflow/models /tf/models

RUN cd /tf/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

RUN cd /tf/models/research && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install .

COPY src /app

WORKDIR /app

CMD python prepare_training_data_set.py && . /train.sh