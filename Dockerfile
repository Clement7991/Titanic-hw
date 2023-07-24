FROM python:3.8.12-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN pip install --upgrade pip
RUN pip install .

CMD uvicorn api.fast:app --host 0.0.0.0 --port 8000
