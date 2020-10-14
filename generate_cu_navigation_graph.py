import sys
import csv
import json
import math
import headset

navigation_graph_file = "cu_navigation_graph.json"

if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print('Usage: %s [session.csv]+')
        print('    For viewing session data see https://wuchlei-thu.github.io/')
        sys.exit(0)

    viewers = 0
    all_views = []
    all_final_tiles = []
    segment_count = 0

    for session in sys.argv[1:]:
        print(session)

        next_view_time = 0
        views = []
        view = 0
        tiles = 0

        with open(session) as file:
            for line in csv.reader(file):
                if 'Timestamp' in line[0]:
                    # skip header
                    continue
                time_ms = float(line[1]) * 1000
                quaternion = tuple([float(s) for s in line[2:6]])
                #print('%s %s' % (str(time_ms), str(quaternion)))

                while time_ms >= next_view_time + headset.segment_ms:
                    # store all views preceding the play time
                    views += [headset.format_view(view)]
                    view = tiles
                    next_view_time += headset.segment_ms

                tiles = headset.get_tiles(quaternion)
                #print('quaternion %s gives tiles %04X' % (str(quaternion), tiles))
                view |= tiles

            #print('time_ms %d  next_view_time %d  headset.segment_ms %d' % (time_ms, next_view_time, headset.segment_ms))
            # store last view
            views += [headset.format_view(view)]

            viewers += 1
            all_views += [views]
            all_final_tiles += [tiles]
            segment_count = max(segment_count, len(views))

    # make sure all sessions have the same length
    for i in range(viewers):
        l = len(all_views[i])
        if l < segment_count:
            all_views[i] += [all_final_tiles[i]] * (segment_count - l)

    all_segment_views = []
    all_segment_views_index = []
    for segment in range(segment_count):
        segment_views = sorted(set([views[segment] for views in all_views]))
        all_segment_views += [segment_views]

        segment_views_index = {}
        for i in range(len(segment_views)):
            segment_views_index[segment_views[i]] = i
        all_segment_views_index += [segment_views_index]

    all_transitions = []
    transitions = [[0] for i in all_segment_views[0]]
    for views in all_views:
        index_from = 0
        index_to = all_segment_views_index[0][views[0]]
        transitions[index_to][index_from] += 1
    all_transitions += [transitions]
    for segment in range(1, segment_count):
        transitions = [[0] * len(all_segment_views[segment - 1]) for i in all_segment_views[segment]]
        for views in all_views:
            index_from = all_segment_views_index[segment - 1][views[segment - 1]]
            index_to = all_segment_views_index[segment][views[segment]]
            transitions[index_to][index_from] += 1
        all_transitions += [transitions]

    # now add final entry to tie up the graph
    last_views = [headset.format_view(0)]
    last_transitions = [[sum(row) for row in all_transitions[-1]]]
    segment_count += 1
    all_segment_views += [last_views]
    all_transitions += [last_transitions]

    graph = []
    for i in range(segment_count):
        entry = {'segment': i}
        entry['views'] = all_segment_views[i]
        entry['transitions'] = all_transitions[i]
        graph += [entry]

    with open(navigation_graph_file, "w") as file:
        #navigation_graph_data = graph
        #json.dump(navigation_graph_data, file)
        #file.write('\n')
        # manual json
        file.write('[\n  ')
        delimit_entries = ''
        for entry in graph:
            file.write(delimit_entries)
            delimit_entries = ',\n  '
            file.write('{\n    "segment": %d,\n' % entry['segment'])
            file.write('    "transitions": [\n      [%s]\n    ],\n' % 
                       '],\n      ['.join([', '.join(['%d' % x for x in row]) for row in entry['transitions']]))
            file.write('    "views": [%s]\n  }' % ', '.join(['"%s"' % v for v in entry['views']]))
        file.write(']\n')
