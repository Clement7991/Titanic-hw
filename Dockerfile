FROM python:3.8.12-alpine

COPY titanic titanic
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install --upgrade pip

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
