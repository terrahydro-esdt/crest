#usage : Source this file to set the python environment and
#environment variables for software execution.  Example
#
#  source /ASTG/sw/crest/cicd/loadEnv.sh crest_cpu
#

instanceRootSearchList=(
   '/ASTG'                                         #AWS,Docker
   '/explore/nobackup/projects/ASTG'               #PRISM
   '/discover/nobackup/projects/TERRAHydro/ASTG'   #Discover
)

echo $instanceRootSearchList

if [[ ! "$1" =~ ^(crest_)*(cpu|gpu) ]]; then
	echo "Please specify cpu or gpu"
else
   if [[ "$1" =~ ^crest_ ]]; then
      sw_proj=$1
   else
      sw_proj="crest_$1"
   fi
   myArch=$(uname -m)
   #if env var ASTG is set & valid, use that.  Otherwise search 
   if [ -z "$ASTG" ] || ! [ -e "$ASTG" ]; then
      for element in "${instanceRootSearchList[@]}"; do
         echo "Looking for ASTG instanceRoot under $element"
         if [ -e "$element" ]; then
            echo "found $element"
            export ASTG=$element
            break #for element
         fi
      done
   fi
   if [ -z "$ASTG" ]; then 
      echo "Failed to identify instanceRoot location for ASTG software, aborting load. "
      echo "Please pre-set the environment variable ASTG to a valid location and re-run"
   else
      export SWROOT="$ASTG/sw"
      distro=$(egrep "^(VERSION_)*ID=" /etc/os-release | sort | awk -F '=' '{print $2}' | sed -e 's/"//g' |  sed -e :a -e '/$/N; s/\n/-/; ta' | sed -e 's/\.[0-9]*$//')
      echo "Loading $sw_proj environment for $distro"
      export PATH=$ASTG/ast_utils/bin:$PATH
      export PYROOT=$ASTG/pyenvs
      if [ -e $PYROOT/$myArch ]; then
         export PYBASE=$PYROOT/$myArch/$distro/production
      elif [ -e $PYROOT/$distro ]; then
         export PYBASE=$PYROOT/$distro/production
      else
         export PYBASE=$PYROOT/production
      fi
      if [ -e $PYBASE ]; then
         source $PYBASE/etc/profile.d/conda.sh
         conda activate $sw_proj
         export PYDIR=$PYBASE/envs/$sw_proj
         export LD_LIBRARY_PATH=$PYDIR/lib
         if [[ "$sw_proj" =~ .*gpu ]]; then
            if [ -e $PYDIR/targets/sbsa-linux ]; then
               export CUDA_PATH=$PYDIR/targets/sbsa-linux
            fi
            if [ -e $PYDIR/bin/mpicc ]; then
               export OMPI_MCA_opal_cuda_support=true
            fi
         fi
      else
         echo "Failed to identify valid python environment, aborting load. "
      fi
   fi
fi

