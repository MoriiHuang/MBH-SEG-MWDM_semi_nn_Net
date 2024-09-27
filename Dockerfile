# Pull the docker image
FROM --platform=linux/amd64 pytorch/pytorch

# FROM python:3.10-slim
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.10-slim-bullseye
RUN (apt-get update) && (apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6)
ENV PYTHONUNBUFFERED 1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user


USER user
WORKDIR /opt/app


# ENV PATH="/home/user/.local/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
RUN python -m piptools sync requirements.txt

# RUN python -m pip install --user torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN python -m pip install --user torch==2.0.0+cu117  torchvision==0.15.1+cu117 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu117

# RUN python -m pip install --user 
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN python -m pip install --no-deps --user nnunetv2


COPY --chown=user:user model /opt/app/model
COPY --chown=user:user intensity-normalization /opt/app/intensity-normalization

RUN cd /opt/app/intensity-normalization && ls -l && python setup.py install --user 

RUN ls -l /opt/app/model
RUN cd /opt/app/model/nnUNet && ls -l && pip install -e .

RUN cd /opt/app/model/MedSAM && ls -l && pip install -e .

RUN cd /opt/app/model/MobileSAM && ls -l && pip install -e .

RUN mkdir input

RUN mkdir output

COPY --chown=user:user Test_script.sh /opt/app/
COPY --chown=user:user Readme.md /opt/app/
# COPY --chown=user:user images/ /opt/app/images/
# COPY --chown=user:user resources/ /opt/app/resources/
# COPY --chown=user:user output/ /opt/app/


ENTRYPOINT [ "sh", "Test_script.sh" ]
