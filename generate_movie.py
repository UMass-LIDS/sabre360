import sys
import csv
import json
import math
import headset

csv_format = 'video_2/size_%d.csv'
csv_id = [0, 1, 2, 3, 4]

movie_file = 'movie360.json'

segment_duration_ms = headset.segment_ms


if __name__ == '__main__':

    segments = None
    tiles = None

    bits_for_quality = []
    for quality in csv_id:
        with open(csv_format % quality) as file:
            bits_for_tiles = []
            for line in csv.reader(file):
                int_line = [int(s) for s in line]
                if segments == None:
                    segments = len(int_line)
                else:
                    assert(segments == len(int_line))
                bits_for_tiles += [int_line]
        if tiles == None:
            tiles = len(bits_for_tiles)
        else:
            assert(tiles == len(bits_for_tiles))
        bits_for_quality += [bits_for_tiles]


    segment_sizes_bits = []
    for s in range(segments):
        bt = []
        for t in range(tiles):
            bq = []
            for q in range(len(csv_id)):
                bq += [bits_for_quality[q][t][s]]
            bt += [bq]
        segment_sizes_bits += [bt]

    bitrates = [0] * len(csv_id)
    for s in range(segments):
        for t in range(tiles):
            for q in range(len(csv_id)):
                bitrates[q] += segment_sizes_bits[s][t][q]
    bitrates = [round(b / (segments * segment_duration_ms)) for b in bitrates]

    movie = {}
    movie['segment_duration_ms'] = segment_duration_ms
    movie['tiles'] = tiles
    movie['bitrates_kbps'] = bitrates
    movie['_comment_'] = 'segment_sizes_bits[segment_index][tile_index][bitrate_index]'
    movie['segment_sizes_bits'] = segment_sizes_bits

    with open(movie_file, 'w') as file:
        #json.dump(movie, file, indent = 2)
        #file.write('\n')
        # manual json

        file.write('{\n')
        file.write('  "segment_duration_ms": %d,\n' % movie['segment_duration_ms'])
        file.write('  "tiles": %d,\n'% movie['tiles'])
        file.write('  "bitrates_kbps": [ %s ],\n' % ', '.join([str(b) for b in movie['bitrates_kbps']]))
        file.write('  "_comment_": "%s",\n' % movie['_comment_'])
        file.write('  "segment_sizes_bits": ')
        segment_text = []
        for s in segment_sizes_bits:
            tile_text = []
            for t in s:
                tile_text += ['[ %s ]' % (', '.join([str(q) for q in t]))]
            segment_text += ['[\n      %s\n    ]' % ',\n      '.join(tile_text)]
        text = '[\n    %s\n  ]' % ',\n    '.join(segment_text)
        file.write(text)
        file.write('\n}\n')
