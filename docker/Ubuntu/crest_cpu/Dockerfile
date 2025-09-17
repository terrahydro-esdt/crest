FROM ubuntu
MAINTAINER ubuntu
SHELL ["/bin/bash", "-c"]
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install curl git libgl-dev libgl1-mesa-dev s3fs rsync wget
RUN DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade
COPY common/* common/.id_git /app/
COPY setup.sh /app/
RUN /app/setup.sh
#CMD ["/app/startup.sh", "crest_cpu"]
#CMD ["/bin/bash"]

