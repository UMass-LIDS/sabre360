import re
import sys
import math
import json
import headset

# TODO: read relevant information from log file and don't use headset module constants

movie_file = 'movie360.json'
default_log_file = 'session.log'

segment_sizes = None
play_period = 0             # playing with at least one tile rendering in view port
incomplete_play_period = 0  # playing with at least one missing tile in view port
stall_period = 0            # playing with no tile rendering in viewport
total_rendered_bits = 0
average_visible_quality = 0
average_visible_bitrate = 0
average_visible_utility = 0
average_quality = 0
average_bitrate = 0
average_utility = 0
average_blank = 0

# blank_penalty gives a utility penalty for blank visible tiles in average_utility
# whereas average_visible_utility ignores blank tiles
# TODO: configurable
# blank_penalty = - utility for highest bitrate
blank_penalty = None

re_timestamp = re.compile('\[([0-9.]+)\] \[([0-9.]+)\]')
re_rendering = re.compile('.*rendering segment=([0-9]+): \((.*)\)')
re_pose = re.compile('.* pose:[^:]*: ([0-9A-F]+)')
re_download_complete = re.compile('.* download complete: segment:([0-9]+) tile:([0-9]+) quality:([0-9]+) ([0-9]+)/([0-9]+)bits')
re_download_abandoned = re.compile('.* download abandoned: segment:([0-9]+) tile:([0-9]+) quality:([0-9]+) ([0-9]+)/([0-9]+)bits')
re_delay_complete = re.compile('.* delay complete: ([0-9.]+)s')
re_action = re.compile('.* action: segment:([0-9]+) tile:([0-9]+) quality:([0-9]+)')

last_calculation_time = 0
rendering_segment = None
rendering_tiles = None
rendering_bits = None
pose = None


def calculate_metrics_up_to(time):
    global segment_sizes

    global last_calculation_time
    global rendering_segment
    global rendering_tiles
    global rendering_bits
    global pose

    global play_period
    global incomplete_play_period
    global stall_period
    global total_rendered_bits
    global average_visible_quality
    global average_visible_bitrate
    global average_visible_utility
    global average_quality
    global average_bitrate
    global average_utility
    global average_blank

    if rendering_segment == None or pose == None:
        last_calculation_time = time
        return

    period = time - last_calculation_time

    visible_tiles = 0
    quality = 0
    bitrate = 0
    utility = 0
    blank_tiles = 0

    for tile in range(tiles):
        (tx, ty, bit) = headset.tile_sequence[tile]
        if bit & pose:
            q = rendering_tiles[tile]
            if q != None:
                visible_tiles += 1
                quality += q
                bitrate += bitrates[q]
                utility += utilities[q]
                total_rendered_bits += rendering_bits[tile]
                rendering_bits[tile] = 0 # avoid double counting
            else:
                blank_tiles += 1


    if visible_tiles > 0:
        play_period += period
        average_visible_quality += (quality / visible_tiles - average_visible_quality) * period / play_period
        average_visible_bitrate += (bitrate / visible_tiles - average_visible_bitrate) * period / play_period
        average_visible_utility += (utility / visible_tiles - average_visible_utility) * period / play_period
        average_quality += ((quality - blank_tiles) / (blank_tiles + visible_tiles) - average_quality) \
            * period / play_period
        average_bitrate += (bitrate / (blank_tiles + visible_tiles) - average_bitrate) * period / play_period
        average_utility += ((utility - blank_penalty * blank_tiles) / (visible_tiles + blank_tiles) - average_utility) \
            * period / play_period
        average_blank += (blank_tiles / (blank_tiles + visible_tiles) - average_blank) * period / (play_period + stall_period)
        if blank_tiles > 0:
            incomplete_play_period += period
        
    else:
        stall_period += period
        average_blank += (1 - average_blank) * period / (play_period + stall_period)
        
    last_calculation_time = time

class OutputTable:

    def __init__(self):
        self.lines = []
        self.lens = [0] * 4

    def add(self, description, value, units, note):
        self.lines += [(description, value, units, note)]
        for i in range(4):
            self.lens[i] = max(self.lens[i], len(self.lines[-1][i]))

    def flush(self):
        format = ('%%%ds: %%%ds %%-%ds    %%s' % tuple(self.lens[:3]))
        for line in self.lines:
            print(format % line)
        self.lines = []
        self.lens = [0] * 4

if __name__ == '__main__':
    if len(sys.argv) == 2:
        log_file = sys.argv[1]
    else:
        log_file = default_log_file

    with open(movie_file) as file:
        raw_manifest = json.load(file)
    tiles = raw_manifest['tiles']
    bitrates = raw_manifest['bitrates_kbps']
    utilities = [math.log(bitrate / bitrates[0]) for bitrate in bitrates]
    blank_penalty = utilities[-1]
    segment_sizes = raw_manifest['segment_sizes_bits']

    all_downloads = {}
    kept_bits = 0
    replaced_bits = 0
    abandoned_bits = 0
    total_delay = 0

    time = None
    with open(log_file) as file:
        for line in file:
            timestamp_match = re_timestamp.match(line)
            if not timestamp_match:
                continue
            time = round(1000 * float(timestamp_match[2]))

            rendering_match = re_rendering.match(line)
            if rendering_match:
                calculate_metrics_up_to(time)
                rendering_segment = int(rendering_match[1])
                rendering_tiles = [int(tile) if tile != '-' else None for tile in rendering_match[2].split(', ')]
                rendering_bits = [0] * len(rendering_tiles)
                for tile in range(len(rendering_tiles)):
                    if rendering_tiles[tile] is not None:
                        rendering_bits[tile] = segment_sizes[rendering_segment][tile][rendering_tiles[tile]]

            pose_match  = re_pose.match(line)
            if pose_match:
                calculate_metrics_up_to(time)
                pose = int(pose_match[1], 16)


            download_complete_match = re_download_complete.match(line)
            if download_complete_match:
                segment = int(download_complete_match[1])
                tile = int(download_complete_match[2])
                quality = int(download_complete_match[3])
                bits = int(download_complete_match[4])
                size = int(download_complete_match[5])
                assert(bits == size)
                assert(size == segment_sizes[segment][tile][quality])
                kept_bits += bits
                key = (segment, tile)
                if key in all_downloads:
                    assert(all_downloads[key] < quality)
                    replaced_bits += segment_sizes[segment][tile][all_downloads[key]]
                all_downloads[key] = quality

            download_abandoned_match = re_download_abandoned.match(line)
            if download_abandoned_match:
                segment = int(download_abandoned_match[1])
                tile = int(download_abandoned_match[2])
                quality = int(download_abandoned_match[3])
                bits = int(download_abandoned_match[4])
                size = int(download_abandoned_match[5])
                assert(bits < size)
                assert(size == segment_sizes[segment][tile][quality])
                abandoned_bits += bits
                key = (segment, tile)
                if key in all_downloads:
                    assert(all_downloads[key] < quality)

            delay_complete_match = re_delay_complete.match(line)
            if delay_complete_match:
                total_delay += float(delay_complete_match[1])


    calculate_metrics_up_to(time)

    output = OutputTable()
    output.add('video_length', '%.3f' % ((play_period + stall_period) / 1000), 's', '')
    output.add('play_period', '%.3f' % (play_period / 1000), 's', 'at least one tile rendering')
    output.add('complete_play_period', '%.3f' % ((play_period - incomplete_play_period) / 1000), 's', 'all tiles rendering')
    output.add('incomplete_play_period', '%.3f' % (incomplete_play_period / 1000), 's', 'at least one tile rendering, at least one tile blank')
    output.add('stall_period', '%.3f' % (stall_period / 1000), 's', 'playing but all tiles blank')
    output.add('play_ratio', '%.6f' % (play_period / (play_period + stall_period)), '', 'play_period / video_length')
    output.add('complete_ratio', '%.6f' % ((play_period - incomplete_play_period) / (play_period + stall_period)), '', 'complete_play_period / video_length')
    output.add('incomplete_ratio', '%.6f' % (incomplete_play_period / (play_period + stall_period)), '', 'incomplete_play_period / video_length')
    output.add('stall_ratio', '%.6f' % (stall_period / (play_period + stall_period)), '', 'stall_ratio / video_length')
    output.add('average_quality', '%.3f' % average_quality, '', 'first average over visible tiles, then over video; blank tiles give -1 quality')
    output.add('average_bitrate', '%.0f' % average_bitrate, 'kbit/s', 'blank tiles give 0 bitrate')
    output.add('average_utility', '%.3f' % average_utility, '', 'blank tiles give utility[max_bitrate]=%.3f penalty' % blank_penalty)
    output.add('average_visible_quality', '%.3f' % average_visible_quality, '', 'average only on rendering tiles')
    output.add('average_visible_bitrate', '%.0f' % average_visible_bitrate, 'kbit/s', '')
    output.add('average_visible_utility', '%.3f' % average_visible_utility, '', '')
    output.add('average_blank', '%.6f' % average_blank, '', 'average of (blank tiles / tiles in view)')
    output.add('total_rendered_bits', '%.3f' % (total_rendered_bits / 1000000), 'Mbit', 'downloaded tiles that were used')
    output.add('download_complete_bits', '%.3f' % (kept_bits / 1000000), 'Mbit', 'download requests that completed')
    output.add('replaced_bits', '%.3f' % (replaced_bits / 1000000), 'Mbit', 'downloaded tiles that were later replaced')
    output.add('abandoned_bits', '%.3f' % (abandoned_bits / 1000000), 'Mbit', 'download requests abandoned before completion')
    output.add('downloaded_bits', '%.3f' % ((kept_bits + abandoned_bits) / 1000000), 'Mbit', 'total download')
    output.add('total_delay', '%.3f' % total_delay, 's', 'time spent not downloading')
    output.flush()
