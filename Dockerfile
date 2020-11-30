FROM nvidia/cuda:11.1.0-base-ubuntu18.04


#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

COPY requirements.txt /app/requirements.txt

RUN ["pip3","install","-r","/app/requirements.txt"]

COPY app/ /app/
WORKDIR /app

CMD ["python3", "torch_train.py"]

