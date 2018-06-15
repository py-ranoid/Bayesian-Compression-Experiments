FROM rombh/gandiva_torch_dev:beta

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

CMD ["python", "lenet.py"]
