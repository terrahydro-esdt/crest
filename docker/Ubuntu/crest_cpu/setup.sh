#!/bin/bash

myOS=`uname -s`
chmod 600 /app/.id_git
export GIT_SSH_COMMAND='ssh -i /app/.id_git -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
echo "Fetching latest MiniForge release..."
wget -nv -O /app/Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
mkdir -p /ASTG/pyenvs
mkdir /ASTG/sw
echo "Setting up conda environment..."
bash /app/Miniforge3-Linux-x86_64.sh -b -p /ASTG/pyenvs/production
/bin/rm -f /app/Miniforge3-Linux-x86_64.sh
cd /ASTG/sw
git clone git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest
/bin/rm -f /app/.id_git
/bin/mv -f /app/.bashrc /home/ubuntu/
/bin/ln -sf /ASTG/sw/crest /crest
#rsync -ax rsync://astg@sinno.net:8873/ASTG/data/crest/ .
source /ASTG/pyenvs/production/etc/profile.d/conda.sh
conda activate
#conda env create --prefix /ASTG/pyenvs/production/envs/crest_cpu -f /ASTG/sw/crest/cicd/environment_${myOS}.yaml
conda env create --prefix /ASTG/pyenvs/production/envs/crest_cpu -f /ASTG/sw/crest/cicd/environment.yaml


