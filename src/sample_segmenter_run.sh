# FOR BEST RESULTS, RUN LIKE THIS:

# /usr/bin/env -i /usr/bin/zsh -f sample_segmenter_run.sh

###########################################################################

# set -x

IR_BASE=$(pwd)
CLASSPATH="${IR_BASE}/*:${IR_BASE}/jai/*"
LIBPATH="${IR_BASE}:${IR_BASE}/jai"
INPUTPATH=scans/linkfarm/20100924_HCC1187/GF4/E06/1
OUTPUTPATHSTEM=newdir/tmp
OUTPUTPATH="${OUTPUTPATHSTEM}.sdc"
SEGPARAM0=7500
SEGPARAM1=1500
SEGPARAM2=1000

ls -ltr $INPUTPATH
rm -rf $( dirname $OUTPUTPATH )
mkdir $( dirname $OUTPUTPATH )

java -Xmx1000M -cp "${CLASSPATH}" -D"java.library.path=${LIBPATH}" run.Segment "${INPUTPATH}" "${OUTPUTPATHSTEM}" 0 0 0 0 0 $SEGPARAM0 $SEGPARAM1 $SEGPARAM2 Centroid

ls -ltr $OUTPUTPATH
