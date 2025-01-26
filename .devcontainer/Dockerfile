FROM mcr.microsoft.com/devcontainers/python:0-3.10

ENV PYTHONUNBUFFERED 1

# [Optional] If your requirements rarely change, uncomment this section to add them to the image.
COPY requirements-CPU.txt /tmp/pip-tmp/
RUN pip3 install --upgrade pip
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-CPU.txt \
   && rm -rf /tmp/pip-tmp