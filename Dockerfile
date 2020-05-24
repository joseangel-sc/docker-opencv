FROM python:3.8.2-slim-buster

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install libglib2.0-0 -y \
    && apt-get install -y libsm6 libxext6 libxrender-dev -y

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
