#!/bin/bash

ASTGapp='crest_cpu'
PYDIR=/opt/ASTG/pyenvs/production

source $PYDIR/etc/profile.d/conda.sh
conda activate $ASTGapp
PROGHOMEDIR=$(cd `dirname $0` && pwd)
crest_root=`dirname $PROGHOMEDIR`
cd $crest_root
echo "Starting crest..."
python host.py
echo "Crest execution exited"
maxRetry=15; loop=1
until [ $loop -ge $maxRetry ];do
	sleep 15
	curl http://localhost:5000/run-dre &> /dev/null
	if [ $? -ne 0 ]; then
		loop=`expr $loop + 1`
	else
		break
	fi
done

