FROM python:3.11

WORKDIR /work

RUN pip install flask==3.0.3

COPY . .
ENV ASSET=/work/assets

ENTRYPOINT [ "flask", "--app", "server", "run", "--host=0.0.0.0" ]
