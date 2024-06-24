ARG PYTHON_BASE=3.10-slim
# build stage
FROM python:$PYTHON_BASE AS builder

# install PDM
RUN pip install -U pdm

# disable update check
ENV PDM_CHECK_UPDATE=false

# copy files
COPY pyproject.toml pdm.lock README.md /project/
COPY src/ /project/src

# install dependencies and project into the local packages directory
WORKDIR /project
RUN pdm install --check --prod --no-editable

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:24.03-py3
FROM $BASE_CONTAINER

LABEL maintainer="Linuzb <levizebulon@gmail.com>"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /project

# retrieve packages from build stage
COPY --from=builder /project/.venv/ /project/.venv
ENV PATH="/project/.venv/bin:$PATH"

COPY . .

ENTRYPOINT ["python", "/project/src/pytorch_ddp_example/__main__.py"]
