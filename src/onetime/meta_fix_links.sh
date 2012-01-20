#!/usr/bin/zsh


ROOT=${1%/}
set -e
[[ -n "$ROOT" ]] || error "Usage: $( basename $0 ) <ROOT_DIR>"

QUITFILE="${0}.QUIT"

fliprow () {
  perl -le "print chr(ord('H') - (ord('$1') - ord('A')))"
}

flipcol () {
  printf '%02d\n' $(( 12 - $1 + 1 ))
}

flipfld () {
  echo $(( ( ( $1 + 1 ) % 4 ) + 1 ))
  # 1 -> 3
  # 2 -> 4
  # 3 -> 1
  # 4 -> 2
}

TMP="$ROOT."'"${$}"'
MV==mv
MKDIR==mkdir
RMDIR==rmdir

echo "ROOT="'"'"$ROOT"'"'
echo "TMP="'"'"$ROOT"'.$$"'
echo
echo "echo "'"TMP directory is ${'"TMP}"'"'
echo "echo"

for w ( $( ls $ROOT ) ) {
  r=${w%??}
  nr=$( fliprow $r )
  c=${w#?}
  nc=$( flipcol $c )
  nw="${nr}${nc}"
  echo
  for f ( $( ls "$ROOT/$w" ) ) {
    ff=${f%.*}
    ext=${f#$ff}
    nf="$( flipfld $ff )$ext"
    [[ -z "$ext" ]] && {
      echo
      echo "${MKDIR} -p "'"${'"TMP}/$nw/$nf"'"'
      for t ( $( ls "$ROOT/$w/$f" ) ) {
  	pl=${t%%_*}
  	rs=${t##*_}
        echo "${MV} -v "'"${'"ROOT}/$w/$f/$t"'"'" "'"${'"TMP}/$nw/$nf/${pl}_${nw}_${nf}_$rs"'"'
      }
      echo "${RMDIR} -v "'"${'"ROOT}/$w/$f"'"'
    } || echo "${MV} -v "'"${'"ROOT}/$w/$f"'"'" "'"${'"TMP}/$nw/$nf"'"'
  }
  echo "${RMDIR} -v "'"${'"ROOT}/$w"'"'
  echo
}

echo "${RMDIR} -v "'"${'"ROOT}"'"'
echo "${MV} -v "'"${'"TMP}"'"'" "'"${'"ROOT}"'"'

# for u ( 0 ) {
#   for w ( $( ls $ROOT ) ) {
#     r=${w%??}
#     nr=$( fliprow $r )
#     c=${w#?}
#     nc=$( flipcol $c )
#     echo
#     for ff ( $( ls "$ROOT/$w" ) ) {
#       f=${ff%.*}
#       ext=${ff#$f}
#       nf="$( flipfld $f )$ext"
#       [[ -z "$ext" ]] && echo && for t ( $( ls "$ROOT/$w/$ff" ) ) {
#     	pl=${t%%_*}
#     	rs=${t##*_}
#         [[ $u == 0 ]] && \
#           echo "mv -v $ROOT/$w/$ff/$t $ROOT/$w/$ff/${pl}_${nr}${nc}_${nf}_$rs" || \
#           echo "# mv -v $ROOT/$w/$ff/${pl}_${nr}${nc}_${nf}_$rs $ROOT/$w/$ff/$t"
#       }
#       [[ $u == 0 ]] && \
#         echo "mv -v $ROOT/$w/$ff $ROOT/$w/$nf" || \
#         echo "# mv -v $ROOT/$w/$nf $ROOT/$w/$ff"
#     }
#     echo
#     [[ $u == 0 ]] && \
#       echo "mv -v $ROOT/$w $ROOT/${nr}${nc}" || \
#       echo "# mv -v $ROOT/${nr}${nc} $ROOT/$w" || \
#   }

#   echo "\n\n\n"
# }
