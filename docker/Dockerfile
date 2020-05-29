o FROM python:3.7-slim

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

EXPOSE 8501

ENTRYPOINT ["./start.sh"]
