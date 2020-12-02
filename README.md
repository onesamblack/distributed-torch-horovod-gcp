# distributed-torch-horovod-gcp
A small example of using GCP horovod and PyTorch for dist training

This is an accompaniment to the articles here:

 - High Performance Distributed Deep Learning with multiple GPUs on Google Cloud Platform — Part 1 [link](https://samsachedina.medium.com/high-performance-distributed-deep-learning-with-multiple-gpus-on-google-cloud-platform-part-1-d7e15604fc34)
 - High Performance Distributed Deep Learning with multiple GPUs on Google Cloud Platform — Part 2 [link](https://samsachedina.medium.com/high-performance-distributed-deep-learning-with-multiple-gpus-on-google-cloud-platform-part-2-5128a6132e03)


## Usage

To run the training script

Step 1: install requirements

```
pip3 install -r requirements.txt
```

Step 2: run the script. This will run on only 1 GPU if available. *you will need at least 1 GPU to run this experiment*

```
python3 app/torch_train.py
```


To run with Horovod (4 GPUs), follow step 1 above and then run:

```
horovodrun -np 4 -H localhost:4 python3 app/torch_train.py 
```

## Building the Image 

To build the image, cd into the repository and run 

```
docker build .
```





