#usage : Source this file to set the python environment and
#environment variables for software execution.  Example
#
#  source /ASTG/sw/crest/cicd/loadEnv.sh crest_cpu
#

failure=0; setPYTHONPATH=1

instanceRootSearchList=(
   'AWS,/ASTG'                                              
   'Discover,/discover/nobackup/projects/TERRAHydro/ASTG'   
   'PRISM,/explore/nobackup/projects/ASTG'                  
)

if [ "$1" = "-swroot" ] || [ "$1" = "--swroot" ] ; then
   SWdir=$2
   shift; shift
else
   SWdir='DEFAULT'
fi
if [ -n "$1" ];then
   if [ "$1" = 'cpu' ]; then
	   conda_envName='OCETRA_cpu'
   elif [ "$1" = 'gpu' ]; then
      conda_envName='OCETRA_gpu'
   else
      conda_envName=$1
   fi
else
	echo "No conda environment name provided, will try OCETRA_gpu and OCETRA_cpu respectively"
   conda_envName='DEFAULT'
fi
ASTG=''
myArch=$(uname -m)
for entry in "${instanceRootSearchList[@]}"; do
   dir_candidate=$(echo $entry | awk -F ',' '{print $2}')
   if [ -e "$dir_candidate" ]; then
      siteName=$(echo $entry | awk -F ',' '{print $1}')
      echo "Matched $siteName environment"
      export ASTG=$dir_candidate
      break 
   fi
done

if [ -z "$ASTG" ]; then 
   echo "Failed to identify instanceRoot location for ASTG software in table below, aborting load. "
else
   if [ "$SWdir" = 'DEFAULT' ]; then
      export SWROOT="$ASTG/sw/kraken"
   else
      export SWROOT=$SWdir
   fi
   distro=$(egrep "^(VERSION_)*ID=" /etc/os-release | sort | awk -F '=' '{print $2}' | sed -e 's/"//g' |  sed -e :a -e '/$/N; s/\n/-/; ta' | sed -e 's/\.[0-9]*$//')
   export PYROOT=$ASTG/pyenvs
   #set PYBASE
   if [ -e $PYROOT/$myArch ]; then
      export PYBASE=$PYROOT/$myArch/$distro/production
   elif [ -e $PYROOT/$distro ]; then
      export PYBASE=$PYROOT/$distro/production
   else
      export PYBASE=$PYROOT/production
   fi
   if [ ! -e $PYBASE ]; then
      echo "Failed to load, unable to find ASTG python root directory. "
      failure=1
   fi
   if [ $failure -eq 0 ] && [ "$conda_envName" = 'DEFAULT' ]; then
      if [ -e $PYBASE/envs/OCETRA_gpu ]; then
         conda_envName='OCETRA_gpu'
      else
         conda_envName='OCETRA_cpu'
      fi
   fi
   echo "Loading $conda_envName environment for $distro"
   if [ $failure -eq 0 ];then
      if [ ! -e $PYBASE/envs/$conda_envName ]; then
         echo "Failed to load python environment $conda_envName, please choose from the following"
         find $PYBASE/envs -mindepth 1 -maxdepth 1 -type d -printf "%f\n"
      else
         source $PYBASE/etc/profile.d/conda.sh
         conda activate $conda_envName
         export PYDIR=$PYBASE/envs/$conda_envName
         export AST_UTILS=$ASTG/ops/ast_utils
         export ASTUTILS=$AST_UTILS
         export ASTUT=$AST_UTILS
         export PATH=$AST_UTILS/bin:$PATH
         export PYTHONPATH=$AST_UTILS/lib:$AST_UTILS/lib/private
         export PYTHONUNBUFFERED=1
         export LD_LIBRARY_PATH=$PYDIR/lib
         export PROJNAME='thdro'
         export THdataBucket='s3://terrahydro-west2-data'
         export THsageBucket='s3://sagemaker-us-west-2-thsmith'
         export THops="$ASTG/ops"
         export THingestDir="$THops/ingest"
         export THstagingDir="$THops/staging"
         if [ -r $AST_UTILS/lib/private/TH_awscreds.sh ]; then
            source $AST_UTILS/lib/private/TH_awscreds.sh
         fi
         case $siteName in
            'AWS')
               export dataMount='/s3/thdro'; export logMount='/efs/thdro'
               export TMPDIR=$ASTG/tmp
            ;;
            'Discover')
               export dataMount=$ASTG; export logMount=$ASTG
            ;;
            'PRISM')
               export dataMount=$ASTG; export logMount=$ASTG
            ;;
         esac
         export THdataDir="$dataMount/data"
         export THsageDir="$dataMount/sagedata"
         export THfcdataDir=$THdataDir/IFS
         export THhcdataDir=$THdataDir/HindCast
         export THlogDir="$logMount/logs"
         export HOSTNAME=`/bin/hostname -s`
         export HOSTFQDN="$HOSTNAME.astg.smce.nasa.gov"
         if [ $setPYTHONPATH -eq 1 ]; then 
            export PYTHONPATH=$PYTHONPATH:$SWROOT/terrahydro:$SWROOT/terrahydro/crest:$SWROOT/terrahydro/terrahydro/data/ingestion/dataset/source:$SWROOT/terrahydro/crest/crest/data/ingestion
         fi
         if [ -e $PYDIR/targets/sbsa-linux ]; then
            export CUDA_PATH=$PYDIR/targets/sbsa-linux
         fi
         if [ -e $PYDIR/bin/mpicc ]; then
            export OMPI_MCA_opal_cuda_support=true
         fi
      fi
   fi
fi

