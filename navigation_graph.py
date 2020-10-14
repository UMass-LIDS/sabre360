import numpy as np
from collections import namedtuple
import sys
import json
import math

import headset


# NavigationGraphEntry corresponds to one segment for cross-user or whole session for single-user.
# views is a list of integers whose bits represent tiles e.g. [ 0x7707, 0x7770, 0xFFF0, 0xFFFF ]
# views_index[view] = index for easy lookup. In e.g. above, view_index[0xFFF0] = 2
# transitions[to, from] = Prob(to = view in segment | from = view in prev_segment) where to and from can be looked up in this.views_index and prev.views_index
# Note that sum(transitions[., from]) = 1.0.
# counts[from] = total number of viewers that had from = view in prev_segment
NavigationGraphEntry = namedtuple('NavigationGraphEntry', 'views views_index transitions counts')


def ng_count_bits_64(v):
    v = ((v & 0xaaaaaaaaaaaaaaaa) >> 1) + (v & 0x5555555555555555)
    v = ((v & 0xcccccccccccccccc) >> 2) + (v & 0x3333333333333333)
    v = ((v & 0xf0f0f0f0f0f0f0f0) >> 4) + (v & 0x0f0f0f0f0f0f0f0f)
    v = ((v & 0xff00ff00ff00ff00) >> 8) + (v & 0x00ff00ff00ff00ff)
    v = ((v & 0xffff0000ffff0000) >> 16) + (v & 0x0000ffff0000ffff)
    v = ((v & 0xffffffff00000000) >> 32) + (v & 0x00000000ffffffff)
    return v

def ng_count_bits(v):
    count = 0
    while v > 0:
        count += ng_count_bits_64(v)
        v >>= 64
    return count

def count_common_tiles(v0, v1):
    return ng_count_bits(v0 & v1)

def ng_view_to_view_vector(graph, weights, view):
    len_views = len(graph.views)
    view_vector = np.zeros(len_views)

    if view in graph.views_index:
        view_vector[graph.views_index[view]] = 1
    else:
        match = None
        index = None
        i = -1
        for v in graph.views:
            i += 1
            if weights[i] <= 0:
                continue
            common_tiles = count_common_tiles(view, v)
            if match is None or common_tiles > match:
                match = common_tiles
                index = [i]
            elif common_tiles == match:
                index += [i]
        assert(index is not None)
        for i in index:
            view_vector[i] = weights[i] # weighted by prior probabilities
        s = sum(view_vector)
        assert(s > 0)
        view_vector /= s

    return view_vector

def ng_view_vector_to_tile_vector(graph, view_vector):
    assert(len(graph.views) == len(view_vector))
    tile_vector = []
    for (tx, ty, bit) in headset.tile_sequence:
        prob = 0
        for i in range(len(view_vector)):
            if graph.views[i] & bit:
                prob += view_vector[i]
        tile_vector += [prob]
    return tile_vector

def check_precision(view, tile_vector):
    precision = 0
    i = 0
    for (tx, ty, bit) in headset.tile_sequence:
        if view & bit:
            precision += tile_vector[i]
        i += 1

    # normalize to maximum 1
    precision /= ng_count_bits(view)
    return precision

class SUNavigationGraph:

    def __init__(self, size = None):
        views = []
        views_index = {}
        if size is None or size < 16:
            size = 16
        transitions = np.zeros([size, size])
        counts = np.zeros(size)
        self.graph = NavigationGraphEntry(views = views, views_index = views_index,
                                          transitions = transitions, counts = counts)

    def view_to_view_vector(self, view):
        if len(self.graph.views) == 0:
            # we have not seen any view information yet
            return None
        # Some new views might not have recorded transitions to other
        # views. The transition matrix has a placeholder such that
        # these views transition to themselves. However, the counts do
        # not account for the placeholder, so we account for it here.
        weights = np.clip(self.graph.counts, 1, None) # clip each count to a minimum 1
        return ng_view_to_view_vector(self.graph, weights, view)

    def view_vector_to_tile_vector(self, view_vector):
        return ng_view_vector_to_tile_vector(self.graph, view_vector)

    def update(self, view, prev_view):
        views = self.graph.views
        views_index = self.graph.views_index
        transitions = self.graph.transitions
        counts = self.graph.counts

        if not prev_view in views_index:
            views += [prev_view]
            views_index[prev_view] = len(views) - 1
        prev_index = views_index[prev_view]

        if not view in views_index:
            views += [view]
            views_index[view] = len(views) - 1
        index = views_index[view]

        l = len(views)
        if l > len(counts):
            grow = math.ceil(0.4 * len(counts)) # double size
            transitions = np.pad(transitions, (0, grow))
            counts = np.pad(counts, (0, grow))
            self.graph = NavigationGraphEntry(views = views, views_index = views_index,
                                              transitions = transitions, counts = counts)
            assert(l <= len(counts))

        count = counts[prev_index]
        transitions[:l, prev_index] *= count / (count + 1)
        transitions[index, prev_index] += 1 / (count + 1)
        counts[prev_index] += 1

        if counts[index] == 0:
            # We do not have any transition from index to some other
            # view. We will set a transition from index to itself, but
            # keep counts[index] == 0 to avoid future inaccuracy.
            # Note that 
            transitions[index, index] = 1

    def predict(self, view_vector):
        len_views = len(self.graph.views)
        if len_views == 0:
            # we have not seen any view information yet
            return None
        return np.matmul(self.graph.transitions[:len_views, :len_views], view_vector)


class CUNavigationGraph:

    def __init__(self, path):

        with open(path) as file:
            raw_navigation_graph = json.load(file)

        self.graph = []
        for segment_entry in raw_navigation_graph:
            views = [int(x, 16) for x in segment_entry['views']]
            views_index = {}
            i = 0
            for v in views:
                views_index[v] = i
                i += 1
            transitions = np.array(segment_entry['transitions'], dtype = float)
            counts = np.sum(transitions, axis = 0) # sum columns
            transitions /= counts # normalize probability to 1
            self.graph += [NavigationGraphEntry(views = views, views_index = views_index,
                                                transitions = transitions, counts = counts)]
        

    def view_to_view_vector(self, segment, view):
       return ng_view_to_view_vector(self.graph[segment], self.graph[segment + 1].counts, view)

    def view_vector_to_tile_vector(self, segment, view_vector):
        return ng_view_vector_to_tile_vector(self.graph[segment], view_vector)

    def predict(self, segment, view_vector):
        return np.matmul(self.graph[segment].transitions, view_vector)


if __name__ == '__main__':
    print('%s is a navigation graph helper module (imports headset module).' % sys.argv[0])
    print('Module provides:')
    print('    count_common_tiles(v0, v1):')
    print('        Returns the number of tiles common to views v0 and v1.')
    print('    check_precision(view, tile_vector):')
    print('        Returns the prediction precision for view')
    print('        where view is the predicted view')
    print('        and tile_vector is a list with a probability per tile.')
    print('    class SUNavigationGraph:')
    print('        Single User Navigation Graph.')
    print('        __init__(self):')
    print('            Initialize.')
    print('        view_to_view_vector(self, view):')
    print('            Converts a single view to a vector of probabilities for all')
    print('            previously-seen views.')
    print('        view_vector_to_tile_vector(self, view_vector):')
    print('            Converts a view_vector to a list with a probability per tile.')
    print('        update(self, view, prev_view):')
    print('            Updates the navigation graph with new entry,')
    print('            recording the transition from prev_view to view.')
    print('        predict(self, view_vector):')
    print('            Returns the predicted view probability vector after')
    print('            having the view probability vector view_vector.')
    print('            Note that this is achieved by multipying the transition matrix')
    print('            by view_vector.')
    print('    class CUNavigationGraph:')
    print('        Cross User Navigation Graph.')
    print('        __init__(self, path):')
    print('            Initialize')
    print('            where path points to a json file containing the navigation graph.')
    print('        view_to_view_vector(self, segment view):')
    print('            Converts a single view to a vector of probabilities for all')
    print('            views that the navigation graph provides for the given segment.')
    print('        view_vector_to_tile_vector(self, segment, view_vector):')
    print('            Converts a view_vector for the givew segment')
    print('            to a list with a probability per tile.')
    print('        predict(self, segment, view_vector):')
    print('            Returns the predicted view probability vector for segment')
    print('            where segment is the segment index of the required prediction')
    print('            and view_vector is the view probability vector for')
    print('            segment index (segment - 1).')
    print('            Note that this is achieved by multipying the transition matrix')
    print('            for segment by view_vector.')
