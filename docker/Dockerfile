FROM python:3.7-slim

ENV http_proxy http://172.16.99.9:3129
ENV https_proxy http://172.16.99.9:3129
ENV HTTP_PROXY=http://172.16.99.9:3129
ENV HTTPS_PROXY=http://172.16.99.9:3129

RUN apt-get update -yqq && \
    apt-get -yqq install binutils git libgtk2.0-dev

RUN mkdir /app
WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

RUN pip install gdown && \
    gdown https://drive.google.com/uc?id=1mEPTm-bT4jTUGHBGbuqwTOq0tp9Womb- && \
    gdown https://drive.google.com/uc?id=1SVa4lw1YZNJ0KKydBguYoD0W2tZUaIXh

RUN git clone https://github.com/Gaarv/CRAFT-pytorch.git && \
    cd CRAFT-pytorch && \
    pip install -r requirements.txt

RUN git clone https://github.com/Gaarv/deep-text-recognition-benchmark.git

CMD ["python3", "app.py"]
