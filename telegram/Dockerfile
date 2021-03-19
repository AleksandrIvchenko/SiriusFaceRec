FROM python:buster

COPY ./requirements.txt /app/requirements.txt

RUN pip install nvidia-pyindex==1.0.4
RUN pip install --requirement /app/requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT ["python"]
CMD ["bot.py"]
