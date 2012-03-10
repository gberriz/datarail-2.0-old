#!/usr/bin/zsh

PATH_TO_H5=$1
# PREFIX=echo; LOG="/proc/$$/fd/1"
PREFIX=; LOG="${0}_${$}.log"; : >$LOG

for a (scans/linkfarm/*) {
  for SUBASSAY (GF CK) {
    ASSAY=$( basename $a )
    [[ -e "$a/${SUBASSAY}1" ]] && \
      $PREFIX bsub -q shared_15m python src/icbp45_makecube.py $PATH_TO_H5 $ASSAY $SUBASSAY
  }
} >>$LOG
