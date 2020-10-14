#!/bin/bash

set -e

output_path="output"
plot_path="plot"
metrics="average_blank average_bitrate average_quality average_utility stall_ratio complete_ratio"

abr_order="baseline bola_p_nr bola_p_r bola_np_nr bola_np_r"

gnuplot_line_colors=(1 2 3 4 5 6 7 8 9 10)
gnuplot_point_types=(1 2 4 6 8 12 13 9 7 5)
gnuplot_points_per_line=10

function abr_name() {
    if [ "$1" = "baseline" ]
    then
        echo Baseline
    elif [ "$1" = "bola_p_nr" ]
    then
        echo "BOLA pb no-rep"
    elif [ "$1" = "bola_p_r" ]
    then
        echo "BOLA pb rep"
    elif [ "$1" = "bola_np_nr" ]
    then
        echo "BOLA no-pb no-rep"
    elif [ "$1" = "bola_np_r" ]
    then
        echo "BOLA no-pb rep"
    else
        echo "$1"
    fi
}

while [ "$#" -ge 2 ]
do
    key="$1"
    shift
    if [ "${key}" = "metric" ]
    then
        metrics="$1"
        shift
    elif [ "${key}" = "output" ]
    then
        output_path="$1"
        shift
    elif [ "${key}" = "plot" ]
    then
        plot_path="$1"
        shift
    else
        echo "Unused parameter \"${key}\"" >&2
        shift
    fi
done

if [ "$#" -eq 1 ]
then
    echo Unused parameter \"$1\" >&2
    shift
fi

mkdir -p "${plot_path}"

table_suffix="-table.out"
cdf_suffix=".cdf"
pdf_suffix=".pdf"

search="${output_path}/*car_0001*viewer_43.out"

buffers=`ls ${output_path}/*.out | cut -s -d '-' -f 2 | sort | uniq`
ids=`ls ${output_path}/*.out | cut -s -d '-' -f 2-3 | sort | uniq`

for id in ${ids}
do
    #break
    echo parsing "${id}"

    table_filename="${plot_path}/${id}${table_suffix}"
    got_headers="no"

    names=`ls ${output_path}/*${id}*.out | cut -s -d '.' -f 1 | cut -s -d '-' -f 2-5 | sort | uniq`
    for name in ${names}
    do
        out_file="${output_path}/session-${name}.out"

        if [ "${got_headers}" = "no" ]
        then
            awk '{ headers[NR] = substr($1, 1, length($1) - 1) } END { delim = "'"session"' "; for (h in headers) { printf("%s%s", delim, headers[h]); delim=" " } print "" }' ${out_file} > ${table_filename}
            got_headers="yes"
        fi

        awk '{ data[NR] = $2 } END { delim = "'"${name}"' "; for (d in data) { printf("%s%s", delim, data[d]); delim=" " } print "" }' ${out_file} >> ${table_filename}
    done
done

for metric in ${metrics}
do
    echo plotting "${metric}"

    for id in ${ids}
    do
        table_filename="${plot_path}/${id}${table_suffix}"
        cdf_filename="${plot_path}/${id}-${metric}${cdf_suffix}"
        awk 'NR==1 { for (i = 1; i <= NF; ++i) {if ($i == "'"${metric}"'") idx = i } } NR>1 { if (idx) value[NR-1] = $idx } END { len = asort(value); for (i = 1; i <= len ; ++i) { print value[i], (i - 1) / (len - 1) } }' ${table_filename} > ${cdf_filename}
    done

    for buffer in ${buffers}
    do
        delimiter=""
        plot=""
        cdf_prefixes=""
        line_count=0
        for abr in ${abr_order}
        do
            cdf_prefix="${plot_path}/${buffer}-${abr}-${metric}"
            cdf_filename="${cdf_prefix}${cdf_suffix}"
            cdf_points_filename="${cdf_prefix}-points${cdf_suffix}"
            if [ -f "${cdf_filename}" ]
            then
                cdf_prefixes="${cdf_prefixes} ${cdf_prefix}"
                title=`abr_name "${abr}"`
                lc="lc ${gnuplot_line_colors[${line_count}]}"
                pt="pt ${gnuplot_point_types[${line_count}]}"
                plot="${plot}${delimiter}\"${cdf_filename}\" with lines ${lc} notitle, \"${cdf_points_filename}\" title \"${title}\" ${lc} ${pt} with linespoints"
                delimiter=", "
                line_count=`expr ${line_count} + 1`
            fi
        done

        line=0
        for cdf_prefix in ${cdf_prefixes}
        do
            cdf_filename="${cdf_prefix}${cdf_suffix}"
            cdf_filename="${cdf_prefix}${cdf_suffix}"
            cdf_points_filename="${cdf_prefix}-points${cdf_suffix}"
            awk 'BEGIN { ppl = '"${gnuplot_points_per_line}"'; line = '"${line}"'; line_count = '"${line_count}"' } { val1[NR] = $1; val2[NR] = $2 } END { for (i = 0; i < ppl; ++i) { idx = int(NR * (i * line_count + line + 1) / (ppl * line_count + 1)) + 1; print val1[idx], val2[idx]; print "" } }' ${cdf_filename} > ${cdf_points_filename}
            line=`expr ${line} + 1`
        done

        metric_space=`tr '_' ' ' <<< ${metric}`
        title="${metric_space} for buffer capacity ${buffer}"
        xlabel=${metric_space}
        ylabel="CDF"
        pdf_filename="${plot_path}/${buffer}-${metric}${pdf_suffix}"
        gnuplot <<EOF
set terminal pdf
set title "${title}"
set xlabel "${xlabel}"
set ylabel "${ylabel}"
set output "${pdf_filename}"
plot ${plot}
set output
EOF

    done
done
