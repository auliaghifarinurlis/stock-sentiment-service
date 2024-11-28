FROM python:3.12-slim as python-base

ENV PORT=4001

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential

WORKDIR /app
COPY requirements.txt /app/

RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT /usr/local/bin/uvicorn main:app --host 0.0.0.0 --port $PORT