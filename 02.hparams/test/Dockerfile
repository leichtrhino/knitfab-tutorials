FROM python:3.11

RUN pip install numpy PyYAML

COPY . .

ENTRYPOINT [ "python", "-u", "test.py" ]
CMD [ "--input-data", "/in/data", "--input-model", "/in/model" ]
