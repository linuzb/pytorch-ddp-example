ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:24.03-py3
FROM $BASE_CONTAINER

LABEL maintainer="Linuzb <levizebulon@gmail.com>"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /src

COPY . .

RUN pip install --no-cache-dir pdm==2.15.3

RUN pdm install

ENTRYPOINT ["python", "/src/src/pytorch_ddp_example/main.py"]
