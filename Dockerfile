FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt update && yes | apt upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt