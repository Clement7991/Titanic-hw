FROM python:3.8.12-buster

COPY titanic titanic
COPY config.py config.py
COPY Makefile Makefile
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV GOOGLE_APPLICATION_CREDENTIALS=/titanic/data/titanic-393415-e3b61a50fb0a.json
EXPOSE 8000

CMD uvicorn titanic.api.fast:app --host 0.0.0.0 --port $PORT
