FROM python:3.11

WORKDIR /work

RUN pip install numpy==1.26.4 pillow==11.0.0

COPY . .

ENTRYPOINT [ "python", "-u", "preprocess.py" ]
CMD [ "--input", "/in/data", "--output-train", "/out/train" ]
