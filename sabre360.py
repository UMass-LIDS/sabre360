# Copyright (c) 2020, authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Units used throughout:
#     size     : bits
#     time     : ms
#     size/time: bits/ms = kbit/s


# Main TODOs:
# 1. Update ABR algorithm
# 2. Test view prediction algorithm
# 3. Support download pipelining for mutiple GETs per segment with non-zero RTT


import argparse
import json
import math
import sys
import string
import os
import numpy as np
from importlib.machinery import SourceFileLoader
from collections import namedtuple

import headset
import navigation_graph as ng

g_debug_cycle = 0
debug_log_level = True

def load_json(path):
    with open(path) as file:
        obj = json.load(file)
    return obj


# segment_duration in ms e.g. 3000
# tiles is number of tiles e.g. 12. Note that tile layout such as 4x3 does not matter here
# bitrates is list of bitrates per tile e.g. [100, 1000] for 100 kbps, 1 Mbps per tile (12 tiles give 1.2, 12 Mbps)
# utilities is list of relative utility per bitrate
# segments[index][tile][quality] is a size in bits
ManifestInfo = namedtuple('ManifestInfo', 'segment_duration tiles bitrates utilities segments')

# A network period lasts "time" ms, provides "bandwidth" kbps, and has "latency" RTT delay
NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')

# play_time: video presentation time for given head pose
# pose: the pose information (TODO: clarify)
PoseInformation = namedtuple('PoseInformation', 'play_time pose')

# delay happens before download
TiledAction = namedtuple('TiledAction', 'segment tile quality delay')

def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)

TiledAction.__str__ = str_tiled_action

# index tile quality: what download is in progress
# size: the size in bits of the download when ready
# downloaded: how many bits have been received by now
# time: total time from request sent until DownloadProgress measured
# time_to_first_bit: time from request sent to first bit (see "latency" in NetworkPeriod)
# abandon: if download was abandoned, then abandon contains information about new action, else abandon is None
DownloadProgress = namedtuple('DownloadProgress', 'segment tile quality size downloaded time time_to_first_bit abandon')

def str_download_progress(self):
    if self.abandon is None:
        abandon = ''
    else:
        abandon = ' abandon_to:(%s)' % str(self.abandon)
    t = self.time / 1000
    ttfb = self.time_to_first_bit / 1000
    tffb = (self.time - self.time_to_first_bit) / 1000
    return ('segment:%d tile:%d quality:%d %d/%dbits %.3f+%.3f=%.3f%ss' %
            (self.segment, self.tile, self.quality, self.downloaded, self.size, ttfb, tffb, t, abandon))

DownloadProgress.__str__ = str_download_progress

class TiledBuffer:

    def __init__(self, segment_duration, tiles):

        self.segment_duration = segment_duration
        self.tiles = tiles
        self.buffers = []
        self.played_segments = 0
        self.played_segment_partial = 0 # measured in ms

    def __str__(self):
        segment0 = self.played_segments
        ret = ('playhead:%.3f first_segment:%d first_segment_offset:%.3fs segment_depth:%d contents:' %
               (self.get_play_head() / 1000, self.played_segments, self.played_segment_partial / 1000, len(self.buffers)))
        for i in range(len(self.buffers)):
            ret += ' segment=%d: (' % (segment0 + i)
            delim = ''
            for t in range(self.tiles):
                q = self.buffers[i][t]
                ret += delim
                delim = ', '
                if q is None:
                    ret += '-'
                else:
                    ret += str(q)
            ret += ')'
        return ret

    def get_played_segments(self):
        return self.played_segments

    def get_played_segment_partial(self):
        return self.played_segment_partial

    def get_buffer_head(self):
        return self.played_segments * self.segment_duration

    def get_play_head(self):
        return self.played_segments * self.segment_duration + self.played_segment_partial

    def get_buffer_depth(self):
        return len(self.buffers)

    def get_buffer_element(self, segment, tile):
        index = segment - self.played_segments
        if not 0 <= index < len(self.buffers):
            return None
        return self.buffers[index][tile]

    def put_in_buffer(self, segment_index, tile_index, quality):
        segment = segment_index - self.played_segments
        if segment < 0:
            # allow case when (segment == 0 and self.played_segment_partial > 0) here,
            # but be careful elsewhere when updating a segment while playing it
            return segment
        grow = segment + 1 - len(self.buffers)
        if grow > 0:
            self.buffers += [[None] * self.tiles for g in range(grow)]
        self.buffers[segment][tile_index] = quality
        return segment

    def play_out_buffer(self, play_time):
        self.played_segment_partial += play_time
        if self.played_segment_partial >= self.segment_duration:
            discard = int(self.played_segment_partial // self.segment_duration)
            del self.buffers[:discard]
            self.played_segments += discard
            self.played_segment_partial %= self.segment_duration


class SessionInfo:

    def __init__(self, manifest, buffer, buffer_size):
        self.manifest = manifest
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.wall_time = 0
        self.presentation_time = 0

    def set_throughput_estimator(self, throughput_estimator):
        self.throughput_estimator = throughput_estimator

    def set_viewport_predictor(self, viewport_predictor):
        self.viewport_predictor = viewport_predictor

    def set_user_model(self, user_model):
        self.user_model = user_model

    def set_log_file(self, log_file):
        self.log_file = log_file

    def get_manifest(self):
        return self.manifest

    def get_buffer(self):
        return self.buffer

    def get_user_model(self):
        return self.user_model

    def get_throughput_estimator(self):
        return self.throughput_estimator

    def get_viewport_predictor(self):
        return self.viewport_predictor

    def get_log_file(self):
        return self.log_file

    def advance_wall_time(self, t):
        self.wall_time += t

    def get_wall_time(self):
        return self.wall_time

class SessionEvents:

    def __init__(self):
        self.play_handlers = []
        self.stall_handlers = []
        self.network_delay_handlers = []
        self.pose_handlers = []

    def add_handler(self, list_of_handlers, handler):
        if handler not in list_of_handlers:
            list_of_handlers += [handler]

    def remove_handler(self, list_of_handlers, handler):
        if handler in list_of_handlers:
            list_of_handlers.remove(handler)

    def trigger_event_0(self, list_of_handlers):
        for handler in list_of_handlers:
            handler()

    def trigger_event_1(self, list_of_handlers, value1):
        for handler in list_of_handlers:
            handler(value1)

    def add_play_handler(self, handler):
        self.add_handler(self.play_handlers, handler)

    def remove_play_handler(self, handler):
        self.remove_handler(self.play_handlers, handler)

    def trigger_play_event(self, time):
        self.trigger_event_1(self.play_handlers, time)

    def add_stall_handler(self, handler):
        self.add_handler(self.stall_handlers, handler)

    def remove_stall_handler(self, handler):
        self.remove_handler(self.stall_handlers, handler)

    def trigger_stall_event(self, time):
        self.trigger_event_1(self.stall_handlers, time)

    def add_network_delay_handler(self, handler):
        self.add_handler(self.network_delay_handlers, handler)

    def remove_network_delay_handler(self, handler):
        self.remove_handler(self.network_delay_handlers, handler)

    def trigger_network_delay_event(self, time):
        self.trigger_event_1(self.network_delay_handlers, time)

    def add_pose_handler(self, handler):
        self.add_handler(self.pose_handlers, handler)

    def remove_pose_handler(self, handler):
        self.remove_handler(self.pose_handlers, handler)

    def trigger_pose_event(self, pose):
        self.trigger_event_1(self.pose_handlers, pose)




class LogFile:
    def __init__(self, session_info, path):
        self.session_info = session_info
        self.fo = open(path, 'w')

    def log_str(self, s):
        self.fo.write('[%.3f] [%.3f] %s\n' % (self.session_info.get_wall_time() / 1000,
                                              self.session_info.get_buffer().get_play_head() / 1000,
                                              s))
        self.fo.flush()

    def log_new_cycle(self, index):
        self.log_str('new cycle: %d' % index)

    def log_tput(self, tput):
        self.log_str('throughput: %.0f kbps' % tput)

    def log_view(self, view):
        self.log_str('tile prediction: %s' % view)

    def log_abr(self, action, from_abandoned = False):
        if not from_abandoned:
            self.log_str('action: %s' % str(action))
        else:
            self.log_str('action (from abandon response): %s' % str(action))

    def log_abr_replace(self, action, old_quality, from_abandoned = False):
        if not from_abandoned:
            self.log_str('action: %s (replace quality:%d)' % (str(action), old_quality))
        else:
            self.log_str('action (from abandon response): %s (replace quality:%d)' % (str(action), old_quality))

    def log_pose(self, pose):
        self.log_str('pose: %s: %s' % (str(pose), headset.format_view(headset.get_tiles(pose))))

    def log_play(self, stall_time, is_startup):
        if is_startup:
            description = 'startup time'
        else:
            description = 'rebuffering time'
        self.log_str('play: %s: %.3fs' % (description, stall_time / 1000))

    def log_stall(self):
        self.log_str('stall')

    def log_delay(self, delay):
        self.log_str('delay complete: %.3fs' % (delay / 1000))

    def log_download(self, progress):
        self.log_str('download complete: %s' % str(progress))

    def log_check_abandon(self, progress):
        self.log_str('check abandon: progress: %s' % str(progress))

    def log_check_abandon_verdict(self, action):
        if action is None:
            a = 'No'
        else:
            a = 'Yes (%s)' % str(action)
        self.log_str('check abandon: verdict: %s' % a)

    def log_abandoned(self, progress):
        self.log_str('download abandoned: %s' % str(progress))

    def log_buffer(self, buffer):
        self.log_str('buffer contents: %s' % buffer)

    def log_rendering(self, segment, tiles):
        self.log_str('rendering segment=%d: (%s)' % (segment, ', '.join([(str(t) if t is not None else '-') for t in tiles])))


class VideoModel:

    def __init__(self, manifest):
        self.manifest = manifest

    def get_manifest(self):
        return self.manifest


class HeadsetModel:

    def __init__(self, session_info, session_events):
        self.session_info = session_info
        self.session_events = session_events
        self.pose_iterator = None
        self.last_pose_info = None
        self.next_pose_info = None
        self.startup_wait = session_info.get_manifest().segment_duration # TODO
        self.is_playing = False

        log_file = self.session_info.get_log_file()
        tile_sequence = headset.tile_sequence
        if len(tile_sequence) > 0:
            (tx, ty, bit) = tile_sequence[0]
            log_file.log_str('First tile: %s: %s' % (self.describe_x_y(tx, ty), headset.format_view(bit)))
        if len(tile_sequence) > 2:
            (txx, tyy, bit) = tile_sequence[1]
            log_file.log_str('Second tile: %s: %s' % (self.describe_x_y_direction(tx, ty, txx, tyy),
                                                      headset.format_view(bit)))
        if len(tile_sequence) > 1:
            (tx, ty, bit) = tile_sequence[-1]
            log_file.log_str('Last tile: %s: %s' % (self.describe_x_y(tx, ty), headset.format_view(bit)))

    def describe_x_y(self, x, y):
        if y == 0:
            describe_y = 'top'
        elif y == headset.tiles_y - 1:
            describe_y = 'bottom'

        if x == 0:
            describe_x = 'left'
        elif x == headset.tiles_x - 1:
            describe_x = 'right'

        return '%s, %s' % (describe_y, describe_x)

    def describe_x_y_direction(self, x, y, xx, yy):
        # not general, only works in one use case
        assert(y == yy or x == xx)
        assert(y != yy or x != xx)

        if y == 0 and yy == y:
            describe_y = 'top'
        elif y == 0 and yy != y:
            describe_y = 'down'
        elif y == headset.tiles_y - 1 and yy == y:
            describe_y = 'bottom'
        elif y == headset.tiles_y - 1 and yy != y:
            describe_y = 'up'

        if x == 0 and xx == x:
            describe_x = 'left'
        elif x == 0 and xx != x:
            describe_x = 'rightward'
        elif x == headset.tiles_x - 1 and xx == x:
            describe_x = 'right'
        elif x == headset.tiles_x - 1 and xx != x:
            describe_x = 'leftward'

        return '%s, %s' % (describe_y, describe_x)

    # returns a list with a weight between 0.0 and 1.0 for each tile
    def get_tiles_for_pose(self, pose):
        tiles = headset.get_tiles(pose)
        ret = []
        for (tx, ty, bit) in headset.tile_sequence:
            if tiles & bit == 1:
                ret += [1.0]
            else:
                ret += [0.0]
        return ret

    # headset model determines whether to rebuffer or not, and also play rate
    def play_for_time(self, time, buffer, time_is_play_time = False):
        wall_time = time

        if self.startup_wait > 0:
            if time_is_play_time:
                time += self.startup_wait
                wall_time += self.startup_wait

            if time >= self.startup_wait:
                stall = self.startup_wait
                self.startup_wait = 0
                time -= stall
                self.is_playing = True
                self.session_events.trigger_stall_event(stall)
            else:
                self.startup_wait -= time
                self.session_events.trigger_stall_event(time)
                return time

        partial_time = 0
        while partial_time < time:
            if self.pose_iterator is None:
                # leave this here - during initialization session_info might not have user model
                self.pose_iterator = self.session_info.get_user_model().get_iterator()
                pose_info = next(self.pose_iterator)
                assert(pose_info is not None)
                self.last_pose_info = pose_info
                self.session_events.trigger_pose_event(self.last_pose_info.pose)
                self.next_pose_info = next(self.pose_iterator)

            head = buffer.get_play_head()

            while self.next_pose_info is not None and head >= self.next_pose_info.play_time:
                self.last_pose_info = self.next_pose_info
                self.session_events.trigger_pose_event(self.last_pose_info.pose)
                self.next_pose_info = next(self.pose_iterator)

            skip = time - partial_time
            # make sure we do not skip over segment boundaries:
            remain_in_segment = self.session_info.get_manifest().segment_duration - buffer.get_played_segment_partial()
            if skip > remain_in_segment:
                skip = remain_in_segment
            # make sure we do not skip over pose changes:
            if self.next_pose_info is not None and skip > self.next_pose_info.play_time - head:
                skip = self.next_pose_info.play_time - head

            partial_time += skip
            self.session_events.trigger_play_event(skip)

        return wall_time


class UserModel:

    def __init__(self, pose_trace):
        self.pose_trace = pose_trace
        self.last_index = 0

    def get_pose(self, time):
        index = self.last_index
        if time < self.pose_trace[index].play_time:
            index = 0
        while index + 1 < len(self.pose_trace) and time >= self.pose_trace[index + 1].play_time:
            index += 1
        if index + 1 < len(self.pose_trace):
            end_time = self.pose_trace[index + 1].play_time
        else:
            end_time = None
        self.last_index = index
        return (self.pose_trace[index].pose, end_time)

    def get_iterator(self):
        last_pose_info = None
        for pose_info in self.pose_trace:
            if last_pose_info is None or pose_info.pose != last_pose_info.pose:
                last_pose_info = pose_info
                yield pose_info
        yield None


class NetworkModel:

    min_progress_size = 12000
    min_progress_time = 50

    def __init__(self, network_trace):

        self.network_total_time = 0
        self.trace = network_trace
        self.index = -1
        self.time_to_next = 0
        self.next_network_period()

    def next_network_period(self):
        self.index += 1
        if self.index == len(self.trace):
            self.index = 0
        self.time_to_next = self.trace[self.index].time


    # return delay time
    def do_latency_delay(self, delay_units):

        total_delay = 0
        while delay_units > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= self.time_to_next:
                total_delay += time
                self.network_total_time += time
                self.time_to_next -= time
                delay_units = 0
            else:
                # time > self.time_to_next implies current_latency > 0
                total_delay += self.time_to_next
                self.network_total_time += self.time_to_next
                delay_units -= self.time_to_next / current_latency
                self.next_network_period()
        return total_delay

    # return download time
    def do_download(self, size):
        total_download_time = 0
        while size > 0:
            current_bandwidth = self.trace[self.index].bandwidth
            if size <= self.time_to_next * current_bandwidth:
                # current_bandwidth > 0
                time = size / current_bandwidth
                total_download_time += time
                self.network_total_time += time
                self.time_to_next -= time
                size = 0
            else:
                total_download_time += self.time_to_next
                self.network_total_time += self.time_to_next
                size -= self.time_to_next * current_bandwidth
                self.next_network_period()
        return total_download_time

    def do_minimal_latency_delay(self, delay_units, min_time):
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_next:
                units = delay_units
                self.time_to_next -= time
                self.network_total_time += time
            elif min_time <= self.time_to_next:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_next -= time
                self.network_total_time += time
            else:
                time = self.time_to_next
                units = time / current_latency
                self.network_total_time += time
                self.next_network_period()
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time
        return (total_delay_units, total_delay_time)

    def do_minimal_download(self, size, min_size, min_time):
        total_size = 0
        total_time = 0
        while size > 0 and (min_size > 0 or min_time > 0):
            current_bandwidth = self.trace[self.index].bandwidth
            if current_bandwidth > 0:
                min_bits = max(min_size, min_time * current_bandwidth)
                bits_to_next = self.time_to_next * current_bandwidth
                if size <= min_bits and size <= bits_to_next:
                    bits = size
                    time = bits / current_bandwidth
                    self.time_to_next -= time
                    self.network_total_time += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_next -= time
                    self.network_total_time += time
                else:
                    bits = bits_to_next
                    time = self.time_to_next
                    self.network_total_time += time
                    self.next_network_period()
            else: # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_next
                    self.network_total_time += time
                    self.next_network_period()
                else:
                    time = min_time
                    self.time_to_next -= time
                    self.network_total_time += time
            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delay(self, time):
        while time > self.time_to_next:
            time -= self.time_to_next
            self.network_total_time += self.time_to_next
            self.next_network_period()
        self.time_to_next -= time
        self.network_total_time += time

    def download(self, size, action, function_check_abandon = None):
        segment = action.segment
        tile = action.tile
        quality = action.quality

        if size <= 0:
            return DownloadProgress(segment = segment, tile = tile, quality = quality,
                                    size = 0, downloaded = 0,
                                    time = 0, time_to_first_bit = 0,
                                    abandon_to_quality = None)

        if not function_check_abandon or (NetworkModel.min_progress_time <= 0 and
                                          NetworkModel.min_progress_size <= 0):
            latency = self.do_latency_delay(1)
            time = latency + self.do_download(size)
            return DownloadProgress(segment = segment, tile = tile, quality = quality,
                                    size = size, downloaded = size,
                                    time = time, time_to_first_bit = latency,
                                    abandon_to_quality = None)

        total_download_time = 0
        total_download_size = 0
        min_time_to_progress = NetworkModel.min_progress_time
        min_size_to_progress = NetworkModel.min_progress_size

        if NetworkModel.min_progress_size > 0:
            latency = self.do_latency_delay(1)
            total_download_time += latency
            min_time_to_progress -= total_download_time
            delay_units = 0
        else:
            latency = None
            delay_units = 1

        abandon_action = None
        while total_download_size < size and abandon_action is None:

            if delay_units > 0:
                # NetworkModel.min_progress_size <= 0
                (units, time) = self.do_minimal_latency_delay(delay_units, min_time_to_progress)
                total_download_time += time
                delay_units -= units
                min_time_to_progress -= time
                if delay_units <= 0:
                    latency = total_download_time

            if delay_units <= 0:
                # don't use else to allow fall through
                (bits, time) = self.do_minimal_download(size - total_download_size,
                                                        min_size_to_progress, min_time_to_progress)
                total_download_time += time
                total_download_size += bits
                # no need to upldate min_[time|size]_to_progress - reset below

            dp = DownloadProgress(segment = segment, tile = tile, quality = quality,
                                  size = size, downloaded = total_download_size,
                                  time = total_download_time, time_to_first_bit = latency,
                                  abandon = None)
            if total_download_size < size:
                abandon_action = function_check_abandon(dp)
                min_time_to_progress = NetworkModel.min_progress_time
                min_size_to_progress = NetworkModel.min_progress_size

        return DownloadProgress(segment = segment, tile = tile, quality = quality,
                                size = size, downloaded = total_download_size,
                                time = total_download_time, time_to_first_bit = latency,
                                abandon = abandon_action)


class ThroughputEstimator:
    def __init__(self, config):
        pass
    def push(self, progress):
        raise NotImplementedError


class Ewma(ThroughputEstimator):

    # for throughput:
    default_half_life = [8000, 3000]

    def __init__(self, config):

        super().__init__(config)

        self.throughput = None
        self.latency = None

        if 'ewma_half_life' in config and config['ewma_half_life'] is not None:
            self.half_life = [h * 1000 for h in config['ewma_half_life']]
        else:
            assert(False)
            self.half_life = Ewma.default_half_life

        # TODO: better?
        self.latency_half_life = [h / min(self.half_life) for h in self.half_life]

        self.throughputs = [0] * len(self.half_life)
        self.weight_throughput = 0
        self.latencies = [0] * len(self.latency_half_life)
        self.weight_latency = 0

    def push(self, progress):

        if progress.time <= progress.time_to_first_bit:
            return

        time = progress.time
        tput = progress.downloaded / (progress.time - progress.time_to_first_bit)
        lat = progress.time_to_first_bit

        for i in range(len(self.half_life)):
            alpha = math.pow(0.5, time / self.half_life[i])
            self.throughputs[i] = alpha * self.throughputs[i] + (1 - alpha) * tput

        for i in range(len(self.latency_half_life)):
            alpha = math.pow(0.5, 1 / self.latency_half_life[i])
            self.latencies[i] = alpha * self.latencies[i] + (1 - alpha) * lat

        self.weight_throughput += time
        self.weight_latency += 1

        tput = None
        lat = None
        for i in range(len(self.half_life)):
            zero_factor = 1 - math.pow(0.5, self.weight_throughput / self.half_life[i])
            t = self.throughputs[i] / zero_factor
            tput = t if tput is None else min(tput, t)  # conservative case is min
            zero_factor = 1 - math.pow(0.5, self.weight_latency / self.latency_half_life[i])
            l = self.latencies[i] / zero_factor
            lat = l if lat is None else max(lat, l) # conservative case is max
        self.throughput = tput
        self.latency = lat

    def get_throughput(self):
        return self.throughput

    def get_latency(self):
        return self.latency


class ViewportPrediction:

    def __init__(self):
        pass

    # returns list of (tile, weight) weighted between 0.0 and 1.0 (list does not necessarily include all tiles)
    def predict_tiles(self, segment_index):
        raise NotImplementedError


class NavigationGraphPrediction(ViewportPrediction):

    def __init__(self, session_info, session_events, navigation_graph_path):
        self.session_info = session_info
        self.session_events = session_events
        self.manifest = session_info.get_manifest()

        self.single_graph = ng.SUNavigationGraph()
        self.cross_graph = ng.CUNavigationGraph(navigation_graph_path)

        self.cur_tiles = 0 # all tiles visible at current time
        self.cur_view = 0 # all tiles visible for some time since last segment transition

        self.prev_view = None # when we finished playing last segment, it had self.prev_view
        self.cur_view_segment = 0 # we are playing inside self.cur_view_segment but not finished
        self.single_memo_prediction = []
        self.cross_memo_prediction = []

        self.choosing_cross_user = True
        self.next_prediction_cross = None
        self.next_prediction_single = None

        self.session_events.add_pose_handler(self.pose_event)
        self.session_events.add_play_handler(self.play_event)

    def pose_event(self, pose):
        self.cur_tiles = headset.get_tiles(pose)
        self.cur_view |= self.cur_tiles

    def play_event(self, time):
        # It is important to use get_played_segment() here and not get_play_head()
        # because otherwise rounding errors might cause some issues elsewhere.
        while self.session_info.get_buffer().get_played_segments() > self.cur_view_segment:
            self.session_info.get_log_file().log_str('view for segment=%d: %s' %
                                                     (self.cur_view_segment, headset.format_view(self.cur_view)))

            # check prediction to compare precision for single-user and cross-user
            if self.cur_view_segment > 1:
                # we can only use single-user prediction after first segment
                single_prediction = self.predict_single(self.cur_view_segment)
                assert(single_prediction is not None)
                single_precision = ng.check_precision(self.cur_view, single_prediction)

                cross_prediction = self.predict_cross(self.cur_view_segment)
                assert(cross_prediction is not None)
                cross_precision = ng.check_precision(self.cur_view, cross_prediction)

                self.choosing_cross_user = cross_precision >= single_precision

                self.session_info.get_log_file().log_str('navigation graph precision: segment=%d, single precision = %.3f, cross precision = %.3f, choosing %s' %
                                                         (self.cur_view_segment, single_precision, cross_precision,
                                                          'cross' if self.choosing_cross_user else 'single'))
            else:
                cross_prediction = self.predict_cross(self.cur_view_segment)
                assert(cross_prediction is not None)
                cross_precision = ng.check_precision(self.cur_view, cross_prediction)
                self.session_info.get_log_file().log_str('navigation graph precision: segment=%d, single precision = -, cross precision = %.3f, choosing %s' %
                                                         (self.cur_view_segment, cross_precision, 'cross'))

            # we get a new view, so we will need to recalculate all predictions
            self.single_memo_prediction = []
            self.cross_memo_prediction = []

            # update single-user graph
            if self.prev_view is not None: # cannot insert startup entry
                self.single_graph.update(self.cur_view, self.prev_view)

            self.prev_view = self.cur_view
            self.cur_view = self.cur_tiles
            self.cur_view_segment += 1


    def predict_single(self, segment):
        memo_index = segment - self.cur_view_segment
        if memo_index < 0:
            # don't predict the past
            return None

        if memo_index < len(self.single_memo_prediction):
            return self.single_memo_prediction[memo_index][1]

        if self.cur_view_segment < 2:
            # we have the first graph after having two segment transitions done
            return None

        if len(self.single_memo_prediction) == 0:
            assert(self.prev_view is not None)
            view_vector = self.single_graph.view_to_view_vector(self.prev_view)
            assert(view_vector is not None)
        else:
            view_vector = self.single_memo_prediction[-1][0]

        while len(self.single_memo_prediction) <= memo_index:
            view_vector = self.single_graph.predict(view_vector)
            tile_vector = self.single_graph.view_vector_to_tile_vector(view_vector)
            self.single_memo_prediction += [(view_vector, tile_vector)]

        return self.single_memo_prediction[memo_index][1]


    def predict_cross(self, segment):
        memo_index = segment - self.cur_view_segment
        if memo_index < 0:
            # don't predict the past
            return None

        if memo_index < len(self.cross_memo_prediction):
            return self.cross_memo_prediction[memo_index][1]

        if len(self.cross_memo_prediction) == 0:
            if self.cur_view_segment == 0:
                # we have not seen any view information yet
                assert(self.prev_view is None)
                view_vector = np.ones(1)
            else:
                assert(self.prev_view is not None)
                view_vector = self.cross_graph.view_to_view_vector(self.cur_view_segment - 1, self.prev_view)
                assert(view_vector is not None)
        else:
            view_vector = self.cross_memo_prediction[-1][0]

        while len(self.cross_memo_prediction) <= memo_index:
            segment = self.cur_view_segment + len(self.cross_memo_prediction)
            view_vector = self.cross_graph.predict(segment, view_vector)
            tile_vector = self.cross_graph.view_vector_to_tile_vector(segment, view_vector)
            self.cross_memo_prediction += [(view_vector, tile_vector)]

        return self.cross_memo_prediction[memo_index][1]


    def predict_tiles(self, segment):
        if self.choosing_cross_user:
            return self.predict_cross(segment)
        else:
            return self.predict_single(segment)


class TiledAbr:

    # TODO: rewrite report_*() to use SessionEvents

    def __init__(self):
        pass
    def get_action(self):
        raise NotImplementedError
    def check_abandon(self, progress):
        return None
    def report_action_complete(self, progress):
        pass
    def report_action_cancelled(self, progress):
        pass
    def report_seek(self, where):
        raise NotImplementedError


class TrivialThroughputAbr(TiledAbr):

    def __init__(self, config, session_info, session_events):
        self.session_info = session_info
        self.session_events = session_events
        self.max_depth = math.floor(session_info.buffer_size / session_info.get_manifest().segment_duration)

    def get_action(self):

        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)
        bitrate_count = len(manifest.bitrates)
        buffer = self.session_info.get_buffer()
        segment0 = buffer.get_played_segments()
        depth = buffer.get_buffer_depth() # how many segments in buffer (one tile in segment enough to count)
        begin = 0
        if buffer.get_played_segment_partial() > 0:
            begin = 1
        end = depth + 1
        end = min(end, self.max_depth + 1) # allow max_depth + 1, but that action requires delay
        end = min(end, segment_count - segment0)
        begin += segment0
        end += segment0

        quality = 0
        tput = self.session_info.get_throughput_estimator().get_throughput()
        if tput is not None:
            if depth <= 1:
                safety_factor = 0.5
            elif depth <= 2:
                safety_factor = 0.6
            elif depth <= 3:
                safety_factor = 0.75
            else:
                safety_factor = 0.9
            for bitrate in manifest.bitrates[1:]:
                if safety_factor * tput >= bitrate:
                    quality += 1

        possible_actions = []
        for segment in range(begin, end):
            for tile in range(manifest.tiles):
                old_quality = buffer.get_buffer_element(segment, tile)
                # Note that when segment == segment0 + depth we are exploring expanding the buffer, old_quality is always None in that case
                if old_quality is None:
                    # add new-download actions
                    possible_actions += [(segment, tile, quality, None)
                                         for quality in range(bitrate_count)]
                else:
                    # add replacement actions
                    possible_actions += [(segment, tile, quality, old_quality)
                                         for quality in range(old_quality + 1, bitrate_count)]
        # filter possible actions by chosen quality level
        possible_actions = [action for action in possible_actions if action[2] == quality]
        best_action = None
        best_score = None
        for action in possible_actions:
            action_score = -action[0] # prefer to download earlier segments
            if action[3] is None:
                action_score += self.max_depth + 2 # give higher priority to new downloads vs replacements

            if best_action is None or action_score > best_score:
                best_action = action
                best_score = action_score

        if best_action is None:
            # happens when we reach end of video
            return None
        else:
            delay = 0
            if best_action[0] == segment0 + self.max_depth:
                delay = manifest.segment_duration - buffer.get_played_segment_partial()
            return TiledAction(segment = best_action[0], tile = best_action[1], quality = best_action[2], delay = delay)

    def check_abandon(self, progress):
        # TODO
        return None

    def report_action_complete(self, progress):
        # TODO
        pass

    def report_action_cancelled(self, progress):
        # TODO
        pass

# TODO: Avoid redundant work
"""
A baseline algorithm where bits are allocated from low to high quality in the order of highest to lowest prediction
probability.

"""

class BaselineAbr(TiledAbr):

    def __init__(self, config, session_info, session_events):
        self.session_info = session_info
        self.session_events = session_events
        self.max_depth = math.floor(session_info.buffer_size / session_info.get_manifest().segment_duration)


    def get_action(self):

        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)
        bitrate_count = len(manifest.bitrates)
        buffer = self.session_info.get_buffer()
        segment0 = buffer.get_played_segments()
        sizes = self.session_info.get_manifest().segments[segment0]
        depth = buffer.get_buffer_depth() # how many segments in buffer (one tile in segment enough to count)
        begin = 0
        if buffer.get_played_segment_partial() > 0:
            begin = 1
        end = depth + 1
        end = min(end, self.max_depth + 1) # allow max_depth + 1, but that action requires delay
        end = min(end, segment_count - segment0)
        begin += segment0
        end += segment0

        quality = 0
        tput = self.session_info.get_throughput_estimator().get_throughput()
        action = None
        segment = None
        tile = None
        blank_tiles = set()

        # select the next segment where at least one tile is None
        for s in range(begin, end):
            bits_used = 0
            for t in range(manifest.tiles):
                if buffer.get_buffer_element(s, t) is None:
                    segment = s
                    blank_tiles.add(t)
                else:
                    bits_used += manifest.segments[s][t][buffer.get_buffer_element(s, t)]
            #if there is something in blank tiles we have found our segment
            if len(blank_tiles) > 0:
                break

        # if tput is None, return None
        if segment is None:
            return None
        #
        # if tput is None:
        #     return TiledAction(segment, None, None, 0)

        tile_probabilities = self.session_info.get_viewport_predictor().predict_tiles(segment)
        if tput is not None:
            qualities = self.allocate_quality(tput * self.session_info.get_manifest().segment_duration - bits_used, segment, tile_probabilities, blank_tiles)
        else:
            qualities = self.allocate_quality(0, segment,
                                              tile_probabilities, blank_tiles)
        for t in blank_tiles:
            if tile is None or tile_probabilities[t] > tile_probabilities[tile]:
                tile = t
        return TiledAction(segment, tile, qualities[tile], 0)

    # what if there is not enough bits to give lowest quality for all tiles?
    def allocate_quality(self, bits, segment, tile_probabilities, blank_tiles):

        manifest = self.session_info.get_manifest()
        buffer = self.session_info.get_buffer()
        sizes = manifest.segments[segment]
        qualities = [0] * manifest.tiles
        curr_tile = 0
        remaining_bits = bits

        for tile in blank_tiles:
                remaining_bits -= sizes[tile][qualities[tile]]
        while remaining_bits > 0:
            found_one = False
            for tile in blank_tiles:
                # find the tile that meets this criteria
                if qualities[tile] < len(manifest.bitrates) - 1:
                    difference = sizes[tile][qualities[tile] + 1] - sizes[tile][qualities[tile]]
                if qualities[tile] < len(manifest.bitrates) - 1 and \
                        tile_probabilities[tile] > 0.0 and \
                        sizes[tile][qualities[tile] + 1] - sizes[tile][qualities[tile]] <= remaining_bits and \
                        (not found_one or
                         (tile_probabilities[tile] / sizes[tile][qualities[tile] + 1]) > \
                         (tile_probabilities[curr_tile] / sizes[curr_tile][qualities[curr_tile] + 1] )):
                    curr_tile = tile
                    found_one = True
            # if no tile is found we are done
            if not found_one:
                break
            qualities[curr_tile] += 1
            remaining_bits -= sizes[curr_tile][qualities[curr_tile]] - sizes[curr_tile][qualities[curr_tile] - 1]

        return qualities

# TODO: design an algorithm worthy of a new name
class ThreeSixtyAbr(TiledAbr):

    # BOLA-based ABR

    def __init__(self, config, session_info, session_events):
        self.know_per_segment_sizes = config['bola_know_per_segment_sizes']
        self.use_placeholder = config['bola_use_placeholder']
        self.allow_replacement = config['bola_allow_replacement']
        self.insufficient_buffer_safety_factor = config['bola_insufficient_buffer_safety_factor']
        self.minimum_tile_weight = config['bola_minimum_tile_weight']

        self.session_info = session_info
        self.session_events = session_events

        manifest = session_info.get_manifest()

        min_buffer_low = 2 * manifest.segment_duration

        if self.use_placeholder:
            self.buffer_low = session_info.buffer_size - manifest.segment_duration * len(manifest.bitrates)
            self.buffer_high = session_info.buffer_size - manifest.segment_duration
            if self.buffer_low < min_buffer_low:
                self.buffer_high += min_buffer_low - self.buffer_low
                self.buffer_low = min_buffer_low
        else:
            self.buffer_low = session_info.buffer_size - manifest.segment_duration * len(manifest.bitrates)
            self.buffer_high = session_info.buffer_size - manifest.segment_duration
            if self.buffer_low < min_buffer_low:
                self.buffer_low = min(min_buffer_low, self.buffer_high / 2)

        # Note: we can use bitrates instead of bits to calculate Vp and gp: scaling the size does not affect them
        self.bitrates = manifest.bitrates
        self.utilities = [math.log(bitrate / self.bitrates[0]) for bitrate in self.bitrates]
        self.average_bits = [bitrate * manifest.segment_duration / manifest.tiles for bitrate in self.bitrates]
        alpha = ((self.bitrates[0] * self.utilities[1] - self.bitrates[1] * self.utilities[0]) /
                 (self.bitrates[1] - self.bitrates[0]))
        self.Vp = (self.buffer_high - self.buffer_low) / (alpha + self.utilities[-1])
        self.gp = ((alpha * self.buffer_high + self.utilities[-1] * self.buffer_low) /
                   (self.buffer_high - self.buffer_low))

        session_events.add_network_delay_handler(self.network_delay_event)
        self.placeholder_buffer = 0

        log_file = self.session_info.get_log_file()
        log_file.log_str('BOLA Vp = %.0f' % self.Vp)
        log_file.log_str('BOLA gp = %.3f' % self.gp)
        log_file.log_str('BOLA target buffer: %.3f-%.3f%s' %
                         (self.buffer_low / 1000, self.buffer_high / 1000,
                          ' (may include buffer expansion)' if self.use_placeholder else ''))

    def network_delay_event(self, time):
        if self.use_placeholder:
            self.placeholder_buffer += time

    def rho(self, bits, utility, buffer_level):
        return (self.Vp * (utility + self.gp) - buffer_level) / bits

    def rho_inc(self, bits, utility, bits_inc, utility_inc, buffer_level, scalable = False):
        # Math outline:
        # Consider qualities having (bits, utility) and (bits + bits_inc, utility + utility_inc).
        # There is some buffer level for which:
        #     rho(bits, utility) == rho(bits + bits_inc, utility + utility_inc).
        # That is:
        #     (Vp * (utility + gp) - buffer_level) / bits
        #         == (Vp * (utility + utility_inc + gp) - buffer_level) / (bits + bits_inc).
        # Solving the above equation gives:
        #     buffer_level = Vp * (utility + gp) - Vp * bits * utility_inc / bits_inc.
        # Assume there exists an incremental download,
        # then we want the following expression to match the above two expressions at buffer_level:
        #     (Vp * u - buffer_level) / bits_inc
        #         == (Vp * (utility + gp) - buffer_level) / bits
        #         == (Vp * (utility + utility_inc + gp) - buffer_level) / (bits + bits_inc).
        # Solving the above equations gives:
        #     u == utility + utility_inc - bits * utility_inc / bits_inc + gp.
        # The objective function for such an incremental download would be:
        #     (Vp * (utility + utility_inc - bits * utility_inc / bits_inc + gp) - buffer_level) / bits_inc.
        # However, incremental downloads are only available for scalable downloads.
        # If scalable downloads are not an option, the denominator must reflect a full new download:
        #     (Vp * (utility + utility_inc + gp - bits * utility_inc / bits_inc) - buffer_level)
        #         / (bits + bits_inc).
        if scalable:
            download_bits = bits_inc
        else:
            download_bits = bits + bits_inc
        return (self.Vp * (utility + utility_inc - bits * utility_inc / bits_inc + self.gp) - buffer_level) / \
            download_bits

    def delay_for_positive_rho(self, utility, buffer_level):
        buffer_target = self.Vp * (utility + self.gp)
        if buffer_level > buffer_target:
            return buffer_level - buffer_target
        else:
            return 0

    def get_action(self):
        global g_debug_cycle

        scalable = False # TODO: update when we start supporting scalable video
        assert(self.use_placeholder or self.placeholder_buffer == 0)
        if debug_log_level and self.use_placeholder:
            self.session_info.get_log_file().log_str('DEBUG BOLA placeholder: %.3f' % (self.placeholder_buffer / 1000))

        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)
        bitrate_count = len(manifest.bitrates)

        buffer = self.session_info.get_buffer()
        buffer_play_head = buffer.get_play_head()

        # check buffer capacity
        buffer_full_level = buffer_play_head + self.session_info.buffer_size
        buffer_full_segment = math.floor(buffer_full_level / manifest.segment_duration)
        buffer_full_delay = (buffer_full_segment + 1) * manifest.segment_duration - buffer_full_level
        assert(0 < buffer_full_delay <= manifest.segment_duration)

        # first segment to consider is first segment in buffer
        begin = buffer.get_played_segments()
        # if first segment started rendering it is too late to update it
        segment0_partial = buffer.get_played_segment_partial()
        if segment0_partial > 0:
            begin += 1

        # last segment to consider (end is exclusive)
        end = min(buffer_full_segment + 1, segment_count)

        throughput = self.session_info.get_throughput_estimator().get_throughput()
        latency = self.session_info.get_throughput_estimator().get_latency()

        assert((throughput is None) == (latency is None))
        safety_throughput = 0 if throughput is None else self.insufficient_buffer_safety_factor * throughput
        safety_latency = 0 if latency is None else latency
        (incomplete_segment, incomplete_tiles, safe_bits_available) = \
            self.get_insufficient_buffer_list(safety_throughput, safety_latency)
        # Note that get_insufficient_buffer_list() does its own buffer level calculation.
        # It is important to not use the placeholder buffer in get_insufficient_buffer_list().

        if debug_log_level:
            self.session_info.get_log_file().log_str('DEBUG Safety: safe_throughput=%.0f safe_latency=%d segment=%s tiles=%s bits=%s'
                                                     % (safety_throughput, safety_latency,
                                                        incomplete_segment if incomplete_segment is not None else '-',
                                                        str(tuple(incomplete_tiles)) if incomplete_tiles is not None else '()',
                                                        round(safe_bits_available) if safe_bits_available is not None else '-'))

        view_predictor = self.session_info.get_viewport_predictor()

        best_objective = None
        best_action = None
        best_action_placeholder_delay = 0
        # We can have (best_objective is None and best_action is not
        # None) when the best action has a negative objective and
        # needs a delay to push its objective up to zero.
        for segment in range(begin, end):
            real_buffer_level = segment * manifest.segment_duration - buffer_play_head
            buffer_level = real_buffer_level + self.placeholder_buffer
            verbose_buffer_level  = '(real=%d*%.3f-%.3f=%.3f)+(placeholder=%.3f)=%.3fs' % (segment, manifest.segment_duration / 1000, buffer_play_head / 1000, real_buffer_level / 1000, self.placeholder_buffer / 1000, buffer_level / 1000)
            assert(buffer_level >= 0)

            min_delay = 0
            if segment == buffer_full_segment:
                min_delay = buffer_full_delay

            # Using (self.minimum_tile_weight + prob) instead of
            # max(self.minimum_tile_weight, prob) does not give a
            # result within the range (0, 1), but (a) it does not
            # matter because the values are only used to multiply
            # objective function values to compare decisions against
            # each other. (b) It still ensures that each tile has a
            # non-zero value and (c) also preserves some difference
            # between tiles with very low probability.
            tile_prediction = [self.minimum_tile_weight + prob for prob in view_predictor.predict_tiles(segment)]

            for tile in range(manifest.tiles):
                quality_bits = manifest.segments[segment][tile] if self.know_per_segment_sizes else self.average_bits
                bits0 = quality_bits[0]

                old_quality = buffer.get_buffer_element(segment, tile)
                if old_quality is not None and not self.allow_replacement:
                    continue
                if old_quality is not None:
                    old_bits = quality_bits[old_quality]
                    old_objective = self.rho(old_bits, self.utilities[old_quality], buffer_level)

                # TODO: While we check all tile possibilities, we can optimize by pruning too far in the future
                #       Note that the following commented block might change behavior in some cases because
                #       prob(segment, tile) can be greater than prob(segment - 1, tile)
                ## if old_quality is None and segment > begin and buffer.get_buffer_element(segment - 1, tile) is None:
                ##     # we already considered adding a new download for this tile
                ##     continue

                for quality in range(0 if old_quality is None else old_quality + 1, bitrate_count):

                    bits = quality_bits[quality]

                    # check insufficient buffer rule
                    if safe_bits_available is not None:
                        safe_bits = safe_bits_available
                        if segment == incomplete_segment and tile in incomplete_tiles:
                            assert(old_quality is None)
                            safe_bits += bits0
                        if bits > safe_bits:
                            continue

                    if old_quality is not None and scalable:
                        download_bits = max(0, bits - old_bits) # do not rely on monotonic bits with quality
                    else:
                        download_bits = bits

                    # Make sure we have enough time to download tile
                    # Exception: do not block tiles in incomplete_tiles
                    if (throughput is not None and download_bits / throughput > real_buffer_level and
                       (segment != incomplete_segment or tile not in incomplete_tiles)):
                        continue

                    objective = self.rho(bits, self.utilities[quality], buffer_level)

                    if old_quality is not None:
                        # replacement download
                        reference_objective = objective
                        objective = self.rho_inc(old_bits, self.utilities[old_quality],
                                                 bits - old_bits, self.utilities[quality] - self.utilities[old_quality],
                                                 buffer_level, scalable = scalable)
                        if objective < 0 or reference_objective < 0 or reference_objective <= old_objective:
                            continue

                    if objective >= 0 and min_delay == 0:
                        objective *= tile_prediction[tile]
                        if best_objective is None or objective > best_objective:
                            best_objective = objective
                            best_action = TiledAction(segment = segment, tile = tile, quality = quality, delay = 0)
                            best_action_placeholder_delay = 0
                    else:
                        # buffer too full for positive objective
                        assert(old_quality is None)
                        # if we already had a non-delay action with an objective > 0, don't attemped delay action
                        if best_objective is None:
                            shrink_buffer = self.delay_for_positive_rho(self.utilities[quality], buffer_level)
                            v = (', buffer_level=%s, delay_for_positive_rho(utility=%.3f, buffer_level=%.3f) = %.3f' %
                                 (verbose_buffer_level, self.utilities[quality], buffer_level / 1000, shrink_buffer / 1000))
                            v += (', shrink_buffer = max(min_delay=%.3fs, delay_for_positive_rho=%.3fs)' %
                                  (min_delay / 1000, shrink_buffer / 1000))
                            shrink_buffer = max(min_delay, shrink_buffer)
                            v += (', placeholder_delay=min(placeholder_buffer=%.3fs, shrink_buffer-min_delay=%.3fs)' %
                                  (self.placeholder_buffer / 1000, (shrink_buffer - min_delay) / 1000))
                            placeholder_delay = min(self.placeholder_buffer, shrink_buffer - min_delay)
                            v += (', delay = (shrink_buffer=%.3fs) - (placeholder_delay=%.3fs))' %
                                  (shrink_buffer / 1000, placeholder_delay / 1000))
                            delay = shrink_buffer - placeholder_delay
                            assert(delay >= min_delay)

                            if (best_action is None or
                                delay < best_action.delay or
                                (delay == best_action.delay and placeholder_delay < best_action_placeholder_delay)):
                                # leave best_objective = None
                                best_action = TiledAction(segment = segment, tile = tile,
                                                          quality = quality, delay = delay)
                                best_action_placeholder_delay = placeholder_delay
                                verbose = v

        if debug_log_level and best_action_placeholder_delay > 0:
            self.session_info.get_log_file().log_str('DEBUG BOLA placeholder shrink by %.3fs %s' %
                                                     (best_action_placeholder_delay / 1000, verbose))

        assert(best_action_placeholder_delay == 0 or best_objective is None)
        self.placeholder_buffer -= best_action_placeholder_delay
        return best_action


    def get_insufficient_buffer_list(self, safety_throughput, latency):
        manifest = self.session_info.get_manifest()
        segment_count = len(manifest.segments)

        buffer = self.session_info.get_buffer()

        # first segment to consider is first segment in buffer
        begin = buffer.get_played_segments()
        # if first segment started rendering it is too late to update it
        if buffer.get_played_segment_partial() > 0:
            begin += 1

        # last segment to consider (end is exclusive)
        end = segment_count

        incomplete_segment = None
        incomplete_tiles = set()
        for segment in range(begin, end):
            for tile in range(manifest.tiles):
                if buffer.get_buffer_element(segment, tile) is None:
                    incomplete_tiles.add(tile)
            if len(incomplete_tiles) > 0:
                incomplete_segment = segment
                break

        if incomplete_segment is None:
            # we have all tiles till the very end
            return (None, None, None)

        time_to_incomplete = incomplete_segment * manifest.segment_duration - buffer.get_play_head()
        assert(time_to_incomplete >= 0)
        bits_to_incomplete = safety_throughput * (time_to_incomplete - latency * (len(incomplete_tiles) + 1))
        if self.know_per_segment_sizes:
            bits_for_tiles = sum([manifest.segments[incomplete_segment][tile][0] for tile in incomplete_tiles])
        else:
            bits_for_tiles = manifest.bitrates[0] * manifest.segment_duration * len(incomplete_tiles) / manifest.tiles
        bits_available = max(0, bits_to_incomplete - bits_for_tiles)

        return (incomplete_segment, incomplete_tiles, bits_available)

    def check_abandon(self, progress):
        if progress.segment <= self.session_info.get_buffer().get_played_segments():
            return self.get_action()
        # TODO
        return None

    def report_action_complete(self, progress):
        # TODO
        pass

    def report_action_cancelled(self, progress):
        # TODO
        pass

class Session:

    def __init__(self, config):
        self.config = config

        raw_manifest = load_json(config['manifest'])
        self.manifest = ManifestInfo(segment_duration = raw_manifest['segment_duration_ms'],
                                     tiles = raw_manifest['tiles'],
                                     bitrates = raw_manifest['bitrates_kbps'],
                                     utilities = raw_manifest['bitrates_kbps'],
                                     segments = raw_manifest['segment_sizes_bits'])
        del raw_manifest

        raw_network_trace = load_json(config['bandwidth_trace'])
        # TODO: Use latency from bandwidth trace.
        #       Currently always setting latency to 5ms to avoid wasting most of the time waiting for RTT.
        self.network_trace = [NetworkPeriod(time = p['duration_ms'],
                                            bandwidth = p['bandwidth_kbps'],
                                            latency = 5)  # TODO: latency = p['latency_ms']
                              for p in raw_network_trace]
        del raw_network_trace

        raw_pose_trace = load_json(config['pose_trace'])
        self.pose_trace = [PoseInformation(play_time = p['time_ms'],
                                           pose = p['quaternion'])
                           for p in raw_pose_trace]
        del raw_pose_trace

        self.buffer_size = config['buffer_size'] * 1000

        self.buffer = TiledBuffer(self.manifest.segment_duration, self.manifest.tiles)
        self.session_info = SessionInfo(self.manifest, self.buffer, self.buffer_size)

        self.log_file = LogFile(self.session_info, config['log_file'])
        self.session_info.set_log_file(self.log_file)

        self.session_events = SessionEvents()
        self.session_events.add_play_handler(self.play_event)
        self.session_events.add_stall_handler(self.stall_event)
        self.session_events.add_pose_handler(self.pose_event)

        self.video_model = VideoModel(self.manifest)
        self.user_model = UserModel(self.pose_trace)
        self.network_model = NetworkModel(self.network_trace)
        self.headset_model  = HeadsetModel(self.session_info, self.session_events)

        self.estimator = Ewma(config)
        self.session_info.set_throughput_estimator(self.estimator)
        self.viewport_prediction = NavigationGraphPrediction(self.session_info, self.session_events,
                                                             config['navigation_graph'])

        self.session_info.set_viewport_predictor(self.viewport_prediction)
        self.session_info.set_user_model(self.user_model)

        self.abr = config['abr'](config, self.session_info, self.session_events)

        self.current_stall_time = 0
        self.did_startup = False


    def play_event(self, time):
        # triggered from HeadsetModel
        if not self.did_startup or self.current_stall_time > 0:
            self.log_file.log_play(self.current_stall_time, not self.did_startup)
            self.current_stall_time = 0
            self.did_startup = True

        while time > 0:
            if self.buffer.get_played_segment_partial() == 0 and self.buffer.get_played_segments() < len(self.manifest.segments):
                segment = self.buffer.get_played_segments()
                tiles = [self.buffer.get_buffer_element(segment, t) for t in range(self.manifest.tiles)]
                self.log_file.log_rendering(segment, tiles)
            do_time = min(time, self.manifest.segment_duration - self.buffer.get_played_segment_partial())
            self.buffer.play_out_buffer(do_time)
            self.session_info.advance_wall_time(do_time)
            time -= do_time


    def stall_event(self, time):
        # triggered from HeadsetModel
        if self.did_startup and self.current_stall_time == 0 and time > 0:
            self.log_file.log_stall()
        self.current_stall_time += time
        self.session_info.advance_wall_time(time)

    def consume_download_time(self, time, time_is_play_time = False):
        # shares self.consumed_download_time with self.run()
        do_time = time - self.consumed_download_time
        assert(do_time >= 0)
        wall_time = self.headset_model.play_for_time(do_time, self.buffer, time_is_play_time = time_is_play_time)
        self.consumed_download_time = time
        return wall_time

    def pose_event(self, pose):
        # triggered from HeadsetModel
        self.log_file.log_pose(pose)

    def check_abandon(self, download_progress):
        # called during self.network_model.download() call inside self.run()
        self.consume_download_time(download_progress.time)
        self.log_file.log_check_abandon(download_progress)
        action = self.abr.check_abandon(download_progress)
        self.log_file.log_check_abandon_verdict(action)
        return action

    def run(self):
        global g_debug_cycle

        video_time = self.manifest.segment_duration * len(self.manifest.segments)
        abandon_action = None
        cycle_index = -1
        tile_prediction_string = None

        while self.session_info.buffer.get_play_head() < video_time:
            cycle_index += 1
            g_debug_cycle = cycle_index
            self.log_file.log_new_cycle(cycle_index)

            if abandon_action is None:
                tput = self.estimator.get_throughput()
                if tput is None: # initial
                    # TODO: better initial estimate?
                    tput = 0
                self.log_file.log_tput(tput)

                view_log_begin = self.buffer.get_played_segments()
                view_log_end = view_log_begin + self.buffer.get_buffer_depth() + 1
                view_log_end = min(view_log_end, len(self.manifest.segments))
                new_tile_prediction_string = ''
                space = ''
                for segment in range(view_log_begin, view_log_end):
                    new_tile_prediction_string += '%ssegment=%d: (' % (space, segment)
                    space = ' '
                    tile_prob = self.viewport_prediction.predict_tiles(segment)
                    delim = ''
                    for p in tile_prob:
                        new_tile_prediction_string += '%s%.4f' % (delim, p)
                        delim = ', '
                    new_tile_prediction_string += ')'
                if new_tile_prediction_string != tile_prediction_string:
                    self.log_file.log_view(new_tile_prediction_string)
                    tile_prediction_string = new_tile_prediction_string

                action = self.abr.get_action()
                old_quality = None if action is None else self.buffer.get_buffer_element(action.segment, action.tile)
                if old_quality is None:
                    self.log_file.log_abr(action)
                else:
                    self.log_file.log_abr_replace(action, old_quality)

            else: # already have action lined up from abandon
                action = abandon_action
                old_quality = self.buffer.get_buffer_element(action.segment, action.tile)
                if old_quality is None:
                    self.log_file.log_abr(action, from_abandoned = True)
                else:
                    self.log_file.log_abr(action, old_quality, from_abandoned = True)
                abandon_action = None


            delay = 0
            if action is None:
                # pause until the end of the current segment
                delay = self.manifest.segment_duration - self.session_info.buffer.get_played_segment_partial()
                #delay = video_time - self.session_info.buffer.get_play_head()
            elif action.delay is not None and action.delay > 0:
                delay = action.delay

            if action is not None:
                buffer_end = (action.segment + 1) * self.manifest.segment_duration
                if delay < buffer_end - self.buffer.get_play_head() - self.buffer_size - 0.001:
                    delay = buffer_end - self.buffer.get_play_head() - self.buffer_size
                    self.log_file.log_str('Buffer full: update delay to %.3fs' % (delay / 1000))

            if delay > 0:
                # shares self.consumed_download_time with self.consume_download_time()
                self.consumed_download_time = 0
                self.network_model.delay(delay)
                self.consume_download_time(delay, time_is_play_time = True)
                self.session_events.trigger_network_delay_event(delay)
                self.log_file.log_delay(delay)

            if action is None:
                continue

            size = self.manifest.segments[action.segment][action.tile][action.quality]
            # shares self.consumed_download_time with self.consume_download_time()
            self.consumed_download_time = 0
            progress = self.network_model.download(size, action, self.check_abandon)
            # Note that self.consume_download_time() may be called multiple times DURING
            # self.network_model.download() call through calls to self.check_abandon().
            self.consume_download_time(progress.time)

            if progress.abandon is None:
                self.estimator.push(progress)
                self.abr.report_action_complete(progress)
                self.buffer.put_in_buffer(progress.segment, progress.tile, progress.quality)
                self.log_file.log_download(progress)
            else:
                self.abr.report_action_cancelled(progress)
                abandon_action = progress.abandon
                self.log_file.log_abandoned(progress)

            self.log_file.log_buffer(self.buffer)


if __name__ == '__main__':
    # TODO: parse arguments for config

    default_config = {}
    default_config['ewma_half_life'] = [4, 1] # seconds
    default_config['buffer_size'] = 5 # seconds
    default_config['manifest'] = 'movie360.json'
    default_config['navigation_graph'] = 'cu_navigation_graph.json'
    default_config['bandwidth_trace'] = 'network.json'
    default_config['pose_trace'] = 'pose_trace.json'
    default_config['log_file'] = 'session.log'

    default_config['bola_know_per_segment_sizes'] = True
    default_config['bola_use_placeholder'] = True
    default_config['bola_allow_replacement'] = True
    default_config['bola_insufficient_buffer_safety_factor'] = 0.5
    default_config['bola_minimum_tile_weight'] = 0.5 / 16

    default_config['abr'] = BaselineAbr

    config = default_config.copy()

    float_args = ['buffer_size', 'bola_insufficient_buffer_safety_factor', 'bola_minimum_tile_weight']
    list_float_args = ['ewma_half_life']
    bool_args = ['bola_know_per_segment_sizes', 'bola_use_placeholder', 'bola_allow_replacement']

    for arg in sys.argv[1:]:
        entry = arg.split('=')
        bad_argument = False
        try:
            if len(entry) != 2 or entry[0] not in config:
                bad_argument = True
            elif entry[0] in float_args:
                config[entry[0]] = float(entry[1])
            elif entry[0] in list_float_args:
                if entry[1][0] != '[' or entry[1][-1] != ']':
                    bad_argument = True
                else:
                    config[entry[0]] = [float(f) for f in entry[1][1].split(',')]
            elif entry[0] in bool_args:
                if entry[1].lower() in ['t', 'true', 'y', 'yes']:
                    config[entry[0]] = True
                elif entry[1].lower() in ['f', 'false', 'n', 'no']:
                    config[entry[0]] = False
                else:
                    bad_argument = True
            elif entry[0] == 'abr':
                if entry[1] == 'TrivialThroughputAbr':
                    config[entry[0]] = TrivialThroughputAbr
                elif entry[1] == 'BaselineAbr':
                    config[entry[0]] = BaselineAbr
                elif entry[1] == 'ThreeSixtyAbr':
                    config[entry[0]] = ThreeSixtyAbr
                else:
                    bad_argument = True
            else:
                config[entry[0]] = entry[1]
        except:
            bad_argument = True
            raise

        if bad_argument:
            print('Bad argument: "%s"' % arg, file = sys.stderr)

    session = Session(config)
    session.run()
