FROM amd64/python:3.8.16-slim

COPY . /app

WORKDIR /app

ENV PIP3_INSTALL="pip3 install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

RUN $PIP3_INSTALL -r /app/requirements.txt

CMD python -u /app/main.py
