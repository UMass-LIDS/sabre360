import sys
import json
import math

config_file = 'headset_config.json'

with open(config_file) as file:
    obj = json.load(file)
    tiles_x = int(obj['tiles_x'])
    tiles_y = int(obj['tiles_y'])
    fov_x_degrees = int(obj['fov_x_degrees'])
    fov_y_degrees = int(obj['fov_y_degrees'])
    segment_ms = int(obj['segment_ms'])

    if tiles_x < 1 or tiles_y < 1:
        print('Headset configuration "%s" has bad "tiles_x" or "tiles_y".' % config_file, file = sys.stderr)
        sys.exit(1)

    if tiles_x == 1 and tiles_y == 1:
        bit_tile_0 = 1
        coords = [(0, 0)]
    else:
        if obj['bit_1_is_tile_0']:
            bit_tile_0 = 1
        else:
            bit_tile_0 = 1 << (tiles_x * tiles_y - 1)
        x_begin = int(obj['tile_0']['x'])
        y_begin = int(obj['tile_0']['y'])
        x_step = int(obj['tile_1']['x']) - x_begin
        y_step = int(obj['tile_1']['y']) - y_begin

        if ((x_step == 0 and y_step == 0) or
            (x_step != 0 and y_step != 0) or
            (tiles_x == 1 and x_step != 0) or
            (tiles_y == 1 and y_step != 0) or
            (x_begin != 0 and x_begin != tiles_x - 1) or
            (y_begin != 0 and y_begin != tiles_y - 1) or
            (y_step == 0 and x_begin == 0 and x_step != 1) or
            (y_step == 0 and x_begin != 0 and x_step != -1) or
            (x_step == 0 and y_begin == 0 and y_step != 1) or
            (x_step == 0 and y_begin != 0 and y_step != -1)):
            print('Headset configuration "%s" has bad "tile_0" or "tile_1".' % config_file, file = sys.stderr)
            sys.exit(1)

        if x_begin == 0:
            xr = range(tiles_x)
        else:
            xr = range(tiles_x - 1, -1, -1)

        if y_begin == 0:
            yr = range(tiles_y)
        else:
            xy = range(tiles_y - 1, -1, -1)

        coords = []
        bit = bit_tile_0
        if y_step == 0:
            for y in yr:
                for x in xr:
                    coords += [(x, y, bit)]
                    if bit_tile_0 == 1:
                        bit <<= 1
                    else:
                        bit >>= 1
        else:
            for x in xr:
                for y in yr:
                    coords += [(x, y, bit)]
                    if bit_tile_0 == 1:
                        bit <<= 1
                    else:
                        bit >>= 1

    tile_sequence = tuple(coords)

    width_x = tiles_x * fov_x_degrees / 360
    width_y = tiles_y * fov_y_degrees / 180

    view_format = '%%0%dX' % ((tiles_x * tiles_y + 3) // 4)

def get_tiles(pose):
    # quaternion equations from doi.org/10.1145/3083187.3083210
    (qx, qy, qz, qw) = pose
    x = 2 * qx * qz + 2 * qy * qw
    y = 2 * qy * qz - 2 * qx * qw
    z = 1 - 2 * qx * qx - 2 * qy * qy

    video_x = math.atan2(-z, x)
    if video_x < 0:
        video_x += 2 * math.pi
    video_x *= tiles_x / (2 * math.pi)

    video_y = math.atan2(math.sqrt(x * x + z * z), y) * tiles_y / math.pi

    left = video_x - width_x / 2
    right = video_x + width_x / 2
    top = video_y - width_y / 2
    bottom = video_y + width_y / 2

    if left < 0:
        left += tiles_x
    if right >= tiles_x:
        right -= tiles_x
    wrap = left > right

    if top < 0:
        top = 0
    if bottom > tiles_y:
        bottom = tiles_y

    tiles = 0
    for (tx, ty, bit) in tile_sequence:
        if ty + 1 >= top and ty <= bottom:
            if not wrap:
                if tx + 1 >= left and tx <= right:
                    tiles |= bit
            else:
                if tx + 1 >= left or tx <= right:
                    tiles |= bit

    return tiles

def format_view(view):
    return view_format % view

if __name__ == '__main__':
    print('%s is a headset helper module.' % sys.argv[0])
    print('See %s for configuration.' % config_file)
    print('Module provides:')
    print('    tiles_x: number of tile columns')
    print('    tiles_y: number of tile rows')
    print('    fov_x_degrees: horizontal field of view in degrees')
    print('    fov_y_degrees: vertical field of view in degrees')
    print('    segment_ms: segment duration in ms')
    print('    tile_sequence: ordered tuple of all possible (x, y, bit), bit -> (x, y)')
    print('    get_tiles(pose):')
    print('        Returns an integer that represents tiles visible from pose')
    print('        where pose is a quaternion tuple (qx, qy, qz, qw).')
    print('    format_view(view):')
    print('        Returns a consistent hexadecimal representation of the view')
    print('        where view is a set of bits representing a set of tiles.')
    print('    Note that a view is the union (logic OR) of all tiles in a segment.')
