import sys
import csv
import json
import math

default_pose_trace_file = "pose_trace.json"

if __name__ == '__main__':

    if not 2 <= len(sys.argv) < 4:
        print('Usage: %s session.csv [pose_trace.json]')
        print('    For viewing session data see https://wuchlei-thu.github.io/')
        sys.exit(0)

    trace = []

    with open(sys.argv[1]) as file:
        for line in csv.reader(file):
            if 'Timestamp' in line[0]:
                # skip header
                continue
            time_ms = round(float(line[1]) * 1000)
            quaternion = tuple([float(s) for s in line[2:6]])
            trace += [{'time_ms': round(time_ms), 'quaternion': quaternion}]

    if len(sys.argv) == 3:
        pose_trace_file = sys.argv[2]
    else:
        pose_trace_file = default_pose_trace_file

    with open(pose_trace_file, "w") as file:
        #json.dump(trace, file, indent = 2)
        #file.write('\n')
        # manual json

        file.write('[\n  ')
        entries = []
        for t in trace:
            quaternion = ', '.join(['%.3f' % q for q in t['quaternion']])
            line = '{ "time_ms": %d, "quaternion": [ %s ] }' % (t['time_ms'], quaternion)
            entries += [line]
        file.write(',\n  '.join(entries))
        file.write('\n]\n')
