#!/bin/bash

ASTG_app=crest_cpu
ASTG_root='/ASTG'
PYDIR=$ASTG_root/pyenvs/production
SWDIR=$ASTG_root/sw/crest
source $PYDIR/etc/profile.d/conda.sh
conda activate $ASTG_app
cd $SWDIR
python host.py

#maxRetry=15; loop=1
#sleep 15
#until [ $loop -ge $maxRetry ];do
#	curl http://localhost:5000/run-dre &> /dev/null
#	if [ $? -ne 0 ]; then
#		loop=`expr $loop + 1`
#	else
#	    break
#	fi
#done


