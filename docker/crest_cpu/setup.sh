#!/bin/bash

myOS=`uname -s`
export GIT_SSH_COMMAND='ssh -i /app/.id_git -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
export RSYNC_PASSWORD=humDRUMPH2
echo "Fetching latest MiniForge release..."
wget -nv -O /app/Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
mkdir -p /opt/pyenvs
mkdir /opt/sw
echo "Setting up conda environment..."
bash /app/Miniforge3-Linux-x86_64.sh -b -p /opt/pyenvs/conda
/bin/rm -f /app/Miniforge3-Linux-x86_64.sh
cd /opt/sw
#git clone git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/terrahydro
#cd terrahydro
git clone git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest
cd crest
rsync -ax --delete rsync://astg@sinno.net:8873/ASTG/data/crest/ data/
source /opt/pyenvs/conda/etc/profile.d/conda.sh
conda activate
conda env create -f /opt/sw/crest/cicd/environment_cpu_${myOS}.yaml

