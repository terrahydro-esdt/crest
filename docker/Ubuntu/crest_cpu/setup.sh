#!/bin/bash

rootDir='/ASTG'
myArch=`uname -m`
PYROOT=$rootDir/pyenvs
SWROOT=$rootDir/sw
chmod 600 /app/.id_git
export GIT_SSH_COMMAND='ssh -i /app/.id_git -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
echo "Fetching latest MiniForge release for $myArch.."
wget -nv -O /app/Miniforge3-Linux-${myArch}.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${myArch}.sh
mkdir -p $PYROOT
mkdir $SWROOT
echo "Setting up conda environment..."
bash /app/Miniforge3-Linux-${myArch}.sh -b -p $PYROOT/production
/bin/rm -f /app/Miniforge3-Linux-${myArch}.sh
cd $SWROOT
git clone git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest
/bin/rm -f /app/.id_git
/bin/mv -f /app/.bashrc /home/ubuntu/
/bin/ln -sf $SWROOT/crest /crest
source $PYROOT/production/etc/profile.d/conda.sh
conda activate
#conda env create --prefix $PYROOT/production/envs/crest_cpu -f /ASTG/sw/crest/cicd/environment_${myArch}.yaml
conda env create --prefix $PYROOT/production/envs/crest_cpu -f /ASTG/sw/crest/cicd/environment.yaml

