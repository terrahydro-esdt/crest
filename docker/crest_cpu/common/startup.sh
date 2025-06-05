#!/bin/bash
#

#fsType=`df -TP /mnt/thdro101 | tail -1 | awk '{print $1}'`
#if [ "$fsType" != 's3fs' ]; then
#   echo "mounting thdro S3 bucket"
#   mount /mnt/thdro101
#fi

ASTGapp=$1
PYDIR=/opt/pyenvs/conda
source $PYDIR/etc/profile.d/conda.sh
conda activate $ASTGapp

case "$ASTGapp" in
   'crest_cpu')
      cd /opt/sw/crest
      python host.py
      ;;
   'autoviz')
      cd /opt/sw/autoviz
      streamlit something something
      ;;
esac

