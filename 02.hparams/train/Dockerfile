FROM python:3.11

RUN pip install numpy PyYAML

COPY . .

ENTRYPOINT [ "python", "-u", "train.py" ]
CMD [ "--input-data", "/in/data", "--input-hparams", "/in/hparams", "--output-model", "/out/model" ]
