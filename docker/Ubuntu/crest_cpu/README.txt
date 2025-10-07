	
	This directory contains files required to build a crest CPU-based
docker container.  Using this as your current working directory, you may
run the commands below to build and run a network-enabled container.  At the
time of this writing (09/16/2025) it is requisite that the 'terrahydro' 
EC2 instance be utilized for the process.  

docker build --no-cache -t crest_cpu:{newVersionId} .
docker tag crest_cpu:{newVersionId} crest_cpu:latest

	For purposes of determining a new version ID, the following command
will list all currently stored crest images

docker image ls -f "reference=crest_*"

