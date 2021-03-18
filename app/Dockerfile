FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./requirements.txt /app/requirements.txt

RUN pip install nvidia-pyindex==1.0.4
RUN pip install --requirement /app/requirements.txt

COPY . /app
