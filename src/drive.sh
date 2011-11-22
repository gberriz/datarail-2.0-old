#!/usr/bin/zsh

# EXAMPLE
# % drive.sh $HOME/IR/scans/linkfarm/20100925_HCC1806 OUTPUTDIR

ASSAYDIR=$1
OUTDIR=$2
set -e

PYTHON="${HOME}/.virtualenvs/base/bin/python";
SRC="${HOME}/IR/src"
MD="${SRC}/dump_well_metadata.py"
DD="${SRC}/dump_well_data.py"
CO="${SRC}/collect_dpf.py"

for z (GF CK) {
  done=
  for k (1 2 3 4) {
    [ "$z" = CK -a "$k" -gt 2 ] && continue
    ppth="$ASSAYDIR/$z$k"
    [ -d "$ppth" ] || continue
    [ -z "$done" ] && done=$ppth || done=$done:$ppth
    (set -x; $PYTHON $MD $ppth;)
    for w (530 685) {
      (set -x; $PYTHON $DD $ppth $w)
      for r (A B C D E F G H) \
        for c ($(seq -w 1 12)) {
        wpth="$ppth/$r$c"
        (set -x; $PYTHON $DD $wpth $w)
      }
      (set -x; $PYTHON $DD $ppth $w)
    }
  }
  [ -z "$done" ] && continue
  zpth="$ASSAYDIR/$z"
  (set -x; $PYTHON $CO $zpth $OUTDIR)
}
