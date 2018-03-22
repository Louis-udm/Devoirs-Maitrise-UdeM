#!/bin/csh -f
#
# felipe@diro
# eval 

if ($#argv != 3) then
  echo "usage: $0 $1 <test> <cand> <ref>"
  exit
endif

set test = $1
set cand = $2
set ref = $3

set n1 = `cat $cand | wc -l`
set n2 = `cat $ref | wc -l`
set n3 = `cat $test | wc -l`

if ($n1 != $n2 || $n1 != $n3 || $n2 != $n3) then
  echo "problème: les fichiers n'ont pas le même nombre de lignes"
  echo "${cand}: $n1 lines"
  echo "${ref}: $n2 lines"
  echo "${test}: $n3 lines"
  exit
endif

paste $cand $ref $test | awk '$3 == "GUESS"' | awk '$1 == $2 { good++ } $1 != $2 { bad++ } END {printf("good: %d bad: %d err: %.2f\n",good,bad,100. * (bad / ((float) (good + bad))))}'
exit
