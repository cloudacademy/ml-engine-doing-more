# ml-engine-doing-more
This file contains text you can copy and paste for the examples in Cloud Academy's _Doing More with Google Cloud Machine Learning Engine_ course.  

### Introduction
[Google Cloud Platform Free Trial](https://cloud.google.com/free)  

### Training a CNN
```
python cnn_mnist.py
```
```
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET=gs://${PROJECT}-ml
DATA_DIR=$BUCKET/data/
REGION=us-central1
JOB=mnist_dist1
cd cnn-mnist
python scripts/create_records.py
gsutil mb -l $REGION $BUCKET
gsutil cp /tmp/data/train.tfrecords $DATA_DIR
gsutil cp /tmp/data/test.tfrecords $DATA_DIR
```
```
gcloud ml-engine jobs submit training $JOB \
    --job-dir $BUCKET/$JOB \
    --runtime-version 1.4 \
    --module-name trainer.task \
    --package-path trainer \
    --region $REGION \
    --scale-tier STANDARD_1 \
    -- \
    --data-dir $DATA_DIR \
    --train-steps 20000
```

### Using TensorBoard
```
tensorboard --logdir=/tmp/mnist_convnet_model &
```
```
tensorboard --port=8080 --logdir=$BUCKET/$JOB
```
```
python cnn_mnist.py
```

### Scale Options
```
cd cnn-mnist
JOB=mnist_custom1
```
```
gcloud ml-engine jobs submit training $JOB \
    --job-dir $BUCKET/$JOB \
    --runtime-version 1.4 \
    --module-name trainer.task \
    --package-path trainer \
    --region $REGION \
    --scale-tier CUSTOM \
    --config custom.yaml \
    -- \
    --data-dir $DATA_DIR \
    --train-steps 20000
```

### Conclusion
[Cloud Machine Learning Engine documentation](https://cloud.google.com/ml-engine/docs)  
