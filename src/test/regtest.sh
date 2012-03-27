#!/usr/bin/zsh

. $ZDOTDIR/.zaliases 2>/dev/null
# LOGDIR=$HOME/local/var/log/$( basename ${0} '.sh' )
LOGDIR=$HOME/local/var/log/$( realpath ${0} | tr '/' '!' )
# PREFIX=echo
# LOG="/proc/$$/fd/1"
PREFIX=; set -e; LOG=$LOGDIR/$( printf %05d $$ )_$( ts )_$( hostname ); set +e
echo $LOG

QUEUE=sorger_15m

PATH_TO_H5=$1
GRP=${2:-"/$USER/reg/$$"}

{
  JN0="$GRP/collect"
  $PREFIX bsub -g $GRP -J $JN0 -q $QUEUE \
    python src/icbp45_collect_confounders.py icbp45_metadata "$PATH_TO_H5"

  JN1="$GRP/mk"
  for a (scans/linkfarm/*) {
    for SUBASSAY (GF CK) {
      ASSAY=$( basename $a )
      JN="$JN1/${ASSAY}__${SUBASSAY}"
      [[ -e "$a/${SUBASSAY}1" ]] && \
        $PREFIX bsub -w "done($JN0)" -g $GRP -J $JN -q $QUEUE \
	  python src/icbp45_makecube.py $PATH_TO_H5 $ASSAY $SUBASSAY
    }
  }

  JN2="$GRP/compareh5"
  $PREFIX bsub -w "done($JN1/*)" -g $GRP -J $JN2 -q $QUEUE \
      python src/test/compareh5.py "$PATH_TO_H5"

  JN3="$GRP/propagate"
  $PREFIX bsub -w "done($JN2)" -g $GRP -J $JN3 -q $QUEUE \
      python src/icbp45_propagate.py "$PATH_TO_H5"

  # diff <(h5dump icbp45.h5 | tail -n +2) <(h5dump "$PATH_TO_H5" | tail -n +2) \
  #   > "$PATH_TO_H5.diff"
  # grep '^< ' "$PATH_TO_H5.diff" | wc -l; grep '^> ' "$PATH_TO_H5.diff" | wc -l
  # # 752
  # # 752

  # grep '^< ' "$PATH_TO_H5.diff" | head -8; echo '...'; \
  #   grep '^> ' "$PATH_TO_H5.diff" | head -8
  # # <             (1,0,0,1,0,0): 121, 2, 11, 4, 5,
  # # <             (1,0,0,1,1,0): 121, 2, 11, 6, 7,
  # # <             (1,0,0,1,2,0): 121, 103, 11, 4, 104,
  # # <             (1,0,0,1,3,0): 121, 103, 11, 6, 105,
  # # <             (1,0,0,1,4,0): 121, 106, 11, 4, 107,
  # # <             (1,0,0,1,5,0): 121, 106, 11, 6, 108,
  # # <             (1,0,0,1,6,0): 121, 109, 11, 4, 110,
  # # <             (1,0,0,1,7,0): 121, 109, 11, 6, 111,
  # # ...
  # # >             (1,0,0,1,0,0): 121, 2, 100, 4, 5,
  # # >             (1,0,0,1,1,0): 121, 2, 100, 6, 7,
  # # >             (1,0,0,1,2,0): 121, 103, 9, 4, 104,
  # # >             (1,0,0,1,3,0): 121, 103, 9, 6, 105,
  # # >             (1,0,0,1,4,0): 121, 106, 9, 4, 107,
  # # >             (1,0,0,1,5,0): 121, 106, 9, 6, 108,
  # # >             (1,0,0,1,6,0): 121, 109, 9, 4, 110,
  # # >             (1,0,0,1,7,0): 121, 109, 9, 6, 111,

  # echo "$0: all OK"
} >>$LOG 2>&1
