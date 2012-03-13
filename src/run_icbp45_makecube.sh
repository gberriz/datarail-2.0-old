#!/usr/bin/zsh

QUEUE=sorger_15m

PATH_TO_H5=$1
GRP=${2:-"/${USER}/mk/$$"}
# PREFIX=echo; LOG="/proc/$$/fd/1"
PREFIX=; LOG=${3:-"${0}_${$}.log"}; : > $LOG

{
  echo $GRP
  for a (scans/linkfarm/*) {
    for SUBASSAY (GF CK) {
      ASSAY=$( basename $a )
      JN="$GRP/$ASSAY/$SUBASSAY"
      [[ -e "$a/${SUBASSAY}1" ]] && \
	echo "bsub -g $GRP -J $JN -q $QUEUE python src/icbp45_makecube.py $PATH_TO_H5 $ASSAY $SUBASSAY"
      [[ -e "$a/${SUBASSAY}1" ]] && \
        $PREFIX bsub -g $GRP -J $JN -q $QUEUE python src/icbp45_makecube.py $PATH_TO_H5 $ASSAY $SUBASSAY
    }
  }

  SENTINEL='No unfinished job found'
  while ( [ 1 ] ) {
    sleep 5
    OUTPUT=$( bjobs -g $GRP |& tail -1 )
    echo $OUTPUT
    if ( [ "$OUTPUT" = "$SENTINEL" ] ) break
  }

  OUTPUT=$( bjobs -d -g $GRP )
  ( echo $OUTPUT | \
    tr -s ' ' | cut -d' ' -f 3 | grep -q EXIT ) && \
    echo $OUTPUT && echo "$0: FAILED" && return 1

  echo "$0: all OK"
} >>$LOG 2>&1

return 0
