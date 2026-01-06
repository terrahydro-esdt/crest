#!/bin/bash

ASTG='/ASTG'
sw_proj='OCETRA_cpu'
distro=$(egrep "^(VERSION_)*ID=" /etc/os-release | sort | awk -F '=' '{print $2}' | sed -e 's/"//g' |  sed -e :a -e '/$/N; s/\n/-/; ta' | sed -e 's/\.[0-9]*$//')
myArch=`uname -m`

PYROOT=$ASTG/pyenvs
SWROOT=$ASTG/sw
PYBASE=$PYROOT/$myArch/$distro/production
PYDIR=$PYBASE/envs/$sw_proj

chmod 600 /app/.id_git
export GIT_SSH_COMMAND='ssh -i /app/.id_git -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
echo "Fetching latest MiniForge release for $myArch.."
wget -nv -O /app/Miniforge3-Linux-${myArch}.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${myArch}.sh
mkdir -p $PYROOT/$myArch/$distro
mkdir $SWROOT

echo "Setting up conda environment..."
bash /app/Miniforge3-Linux-${myArch}.sh -b -p $PYBASE
/bin/rm -f /app/Miniforge3-Linux-${myArch}.sh
cd $SWROOT
git clone -b env_dev git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest
/bin/rm -f /app/.id_git
/bin/mv -f /app/.bashrc /home/ubuntu/
/bin/ln -sf $SWROOT/crest /crest
source $PYBASE/etc/profile.d/conda.sh
conda activate
echo "conda env create --prefix $PYDIR -f $SWROOT/crest/cicd/environment_cpu_${myArch}.yaml"
conda env create --prefix $PYDIR -f $SWROOT/crest/cicd/environment_cpu_${myArch}.yaml


