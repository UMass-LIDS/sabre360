#!/bin/bash

set -e

abrs="baseline bola_p_nr bola_p_r bola_np_nr bola_np_r"
buffers="5"
networks=4Glogs/*.json
viewers=`seq 43 48`
start_at="1"
output_path="output"

while [ "$#" -ge 2 ]
do
    key="$1"
    shift
    if [ "${key}" = "abr" ]
    then
        abrs="$1"
        shift
    elif [ "${key}" = "buffer" ]
    then
        buffers="$1"
        shift
    elif [ "${key}" = "network" ]
    then
         networks="$1"
         shift
    elif [ "${key}" = "viewer" ]
    then
        viewers="$1"
        shift
    elif [ "${key}" = "start" ]
    then
        start="$1"
        shift
    elif [ "${key}" = "output" ]
    then
         output_path="$1"
         shift
    else
        echo "Unused parameter \"${key}\"" >&2
    fi
done
if [ "$#" -eq 1 ]
then
    echo Unused parameter \"$1\" >&2
    shift
fi

mkdir -p "${output_path}"


current_at=0

for viewer in ${viewers}
do
    for network in ${networks}
    do
        for buffer in ${buffers}
        do
            for abr in ${abrs}
            do
                current_at=`expr ${current_at} + 1`
                if [ ${current_at} -lt ${start_at} ]
                then
                    continue
                fi

                trace=`basename ${network} .json`

                if [ ${abr} = "baseline" ]
                then
                    abr_param='abr=BaselineAbr'
                elif [ ${abr} = "bola_p_nr" ]
                then
                    abr_param='abr=ThreeSixtyAbr bola_allow_replacement=no bola_use_placeholder=yes'
                elif [ ${abr} = "bola_p_r" ]
                then
                    abr_param='abr=ThreeSixtyAbr bola_allow_replacement=yes bola_use_placeholder=yes'
                elif [ ${abr} = "bola_np_nr" ]
                then
                    abr_param='abr=ThreeSixtyAbr bola_allow_replacement=no bola_use_placeholder=no'
                elif [ ${abr} = "bola_np_r" ]
                then
                    abr_param='abr=ThreeSixtyAbr bola_allow_replacement=yes bola_use_placeholder=no'
                fi

                filename="${output_path}/session-${buffer}s-${abr}-${trace}-viewer_${viewer}"
                param="${abr_param} buffer_size=${buffer} bandwidth_trace=${network} pose_trace=video_2/viewer_${viewer}.json log_file=${filename}.log"

                echo ${current_at} ${buffer}s ${abr} ${trace} viewer_${viewer}

                python3 sabre360.py ${param}
                python3 calculate_metrics.py ${filename}.log > ${filename}.out

            done
        done
    done
done
