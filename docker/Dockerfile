FROM nvcr.io/nvidia/clara/monai-toolkit:2.5

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ARG USER_ID
ARG GROUP_ID
USER root

EXPOSE 8888

COPY requirements.txt .
RUN apt-get update
RUN apt-get -y install sudo

RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev
RUN sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository universe
RUN sudo apt-get update

RUN pip install -r requirements.txt