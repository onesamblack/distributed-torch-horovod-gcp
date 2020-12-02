FROM nvidia/cuda:11.1-runtime-ubuntu18.04


#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install cmake

COPY requirements.txt /app/requirements.txt

RUN ["pip3","install","-r","/app/requirements.txt"]

COPY app/ /app/
WORKDIR /app

CMD ["horovodrun","-np", 4, "-H", "localhost:4", "app/torch_train.py"]

