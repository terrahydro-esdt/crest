FROM crest_cpu:latest

SHELL ["/bin/bash", "-c"]

COPY docker/common/.id_git /app/
RUN /bin/chmod 600 /app/.id_git
COPY docker/common/.gitconfig /home/ubuntu/
COPY docker/common/startup.sh /app/

ENV GIT_SSH_COMMAND "ssh -i /app/.id_git -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
WORKDIR /ASTG/sw/crest/
RUN /usr/bin/git pull
RUN /bin/rm -f /app/.id_git
RUN /app/startup.sh
