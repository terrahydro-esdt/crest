FROM debian
MAINTAINER admin
SHELL ["/bin/bash", "-c"]
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install apt-utils
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install wget git rsync libgl-dev libgl1-mesa-dev s3fs
RUN DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
COPY common/* common/.id_git /app/
COPY setup.sh /app/
RUN /app/setup.sh
CMD ["/app/startup.sh", "crest_cpu"]

