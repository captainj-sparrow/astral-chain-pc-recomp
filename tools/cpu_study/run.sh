#!/bin/sh
# usage: run.sh <exe name> <tag> <seconds> <wall limit s> [KEY=VALUE ...]
# One bounded scripted run of build/cpu-study/v101/<exe name> on build/cpu-study/rec-cam.txt (the recording
# build/barriers/rec.txt: Harmony Square free play from ~35 s, plus a slow camera turn from 38 s), from a fresh copy
# of build/barriers/prof-start, in a 1920x1080 window. Log: build/cpu-study/logs/<tag>.log.
# The game is killed by path if it outlives the wall limit.
R=$(cd "$(dirname "$0")/../.." && pwd)
B=$R/build/cpu-study   # copy the exes into $B/v101 and this script's helpers stay in tools/cpu_study
exe=$1; tag=$2; secs=$3; wall=$4; shift 4
mkdir -p $B/logs $B/shots/$tag
log=$B/logs/$tag.log
prof=$B/prof-$tag
rm -rf "$prof" && cp -r $R/build/barriers/prof-start "$prof" || exit 1
cd $B/v101 || exit 1
env AC_ASSUME_FOCUS=1 AC_INPUT=none AC_INPUT_SCRIPT=${SCRIPT:-$B/rec-cam.txt} AC_BACKGROUND_WINDOW=1 \
    AC_WINDOW_SIZE=1920x1080 AC_FULLSCREEN=0 AC_SHOTS=w40-76:18 AC_SHOTS_DIR=$B/shots/$tag \
    AC_TRACE=${TRACE:-perf,sched} "$@" \
    timeout "$wall" ${AFF:+$R/.venv/Scripts/python $R/tools/cpu_study/aff.py $AFF ${AFFDELAY:-36}} ./$exe --profile "$prof" --seconds "$secs" > "$log" 2>&1
rc=$?
powershell -NoProfile -Command "Get-Process | Where-Object { \$_.Path -like '*build\cpu-study\v101\*' } | ForEach-Object { Stop-Process -Id \$_.Id -Force; 'killed ' + \$_.Id }"
rm -rf "$prof"
echo "run.sh: $tag rc=$rc; fatal/device lost lines: $(grep -c 'FATAL\|device lost' "$log")"
