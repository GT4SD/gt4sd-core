# -*- coding:utf-8 -*-
"""
ChemGraphFragment.py

Package for IBM Molecule Generation Experience

MIT License

Copyright (c) 2022 International Business Machines Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .ChemGraph import *
from .ChemGraphLabeling import *

from collections import Counter, defaultdict
from operator import itemgetter
import copy

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def find_construction_path(graph, labeling_func=None):
    """Get a list of vertices in a graph by sorting them in an order of addition to the graph
    in the graph generation algorithm.

    Args:
        graph (AtomGraph): a graph
        labeling_func (func): a labeling function

    Returns:
        list: a list of vertex ordered by addition to the graph
    """
    # make a copy of graph
    graph_copy = copy.deepcopy(graph)
    # make a map to original vertex
    v_map = {}
    for index in range(0, len(graph.vertices)):
        v_map[graph_copy.vertices[index]] = graph.vertices[index]
    path = find_construction_path0(graph_copy, None, labeling_func)
    return [v_map[v] for v in path]


def find_construction_path0(graph, const_pos, labeling_func=None):
    """Get a list of vertices ordered by addition to the graph with a vertex the last vertex is connected to.

    Args:
        graph (AtomGraph): a graph
        const_pos (AtomVertex, None): a vertex the last vertex is connected to
        labeling_func (func): a labeling function

    Returns:
        list: a list of vertex ordered by addition to the graph
    """
    if len(graph.vertices) == 0:
        return []
    elif len(graph.vertices) == 1:
        return [graph.pop_vertex()]
    if labeling_func is not None:
        labeling = labeling_func(graph)
    else:
        labeling = ChemGraphLabeling(graph.vertices)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('const_path:%s', graph.to_string())
        logger.debug('const_path:%s', graph.to_string_label())
    # assign max_label among vertex orbit to construction position
    if const_pos is not None:
        # check if const_pos is initial connection to a ring
        max_label = -1
        max_index = 0
        for index in labeling.automorphism.orbit(const_pos.index):
            if graph.vertices[index].label >= max_label:
                max_label = graph.vertices[index].label
                max_index = index
        if max_label > const_pos.label:
            # swap label with const_pos and vertices[max_index]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('const_pos:swap %d %d', const_pos.index, max_index)
            graph.vertices[max_index].label = const_pos.label
            const_pos.label = max_label
    zero_label = None
    for v in graph.vertices:
        if v.label == 0:
            zero_label = v
            break
    if len(graph.vertices) > 2:
        min_label = len(graph.vertices)
        min_index = 0
        for index in labeling.automorphism.orbit(zero_label.index):
            if graph.vertices[index].edges[0].end.label <= min_label:
                min_label = graph.vertices[index].edges[0].end.label
                min_index = index
        if min_label < zero_label.edges[0].end.label:
            # swap label with zero_label and vertices[min_index]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('zero_label:swap %d %d', zero_label.index, min_index)
            zero_label.label = graph.vertices[min_index].label
            zero_label = graph.vertices[min_index]
            zero_label.label = 0
    # remove zero_labeled vertex
    new_const_pos = zero_label.edges[0].end
    zero_label_index = zero_label.index
    graph.pop_vertex(zero_label)
    vertex_map = dict()
    if zero_label.ring_atom():
        # ring is broken
        original_vertices = [v for v in graph.vertices]
        graph.update_atom_in_ring()
        for origin, replace in zip(original_vertices, graph.vertices):
            if new_const_pos == origin:
                new_const_pos = replace
            vertex_map[replace] = origin
    # renumber vertex index
    for v in graph.vertices:
        if v.index > zero_label_index:
            v.index = v.index - 1
    path = find_construction_path0(graph, new_const_pos)
    if zero_label.ring_atom():
        path = [vertex_map[v] for v in path]
    path.append(zero_label)
    return path


class ChemFragment(object):
    """Fragment of molecular graph, whose occurrences are counted during the graph generation.
    If the fragment include dummy atom symbol (*), the occurrences are counted by matching exactly
    the number of edges from the atom vertex. Therefore, the occurrence does not increase monotonically
    during the graph generation.

    Attributes:
        graph (AtomGraph): a graph of the fragment
        atom_count (Counter): counter of atoms in a fragment
        labeling (ChemGraphLabeling): labeling of a fragment
        exact_match (bool): flag of exact match mode
        in_ring (bool): flag if a fragment matches also to atoms in a ring
        wild_card_vertices (list): a list of wild_card(*) vertices in a fragment
        root_vertex (AtomVertex): root atom of fragment graph
        root_group (defaultdict): a dictionary of root vertices by radius
    """

    def __init__(self, graph, labeling_func=None):
        """Constructor of fragment graph.

        Args:
            graph (AtomGraph): a graph representing a fragment to count
        """
        self.graph = graph
        self.atom_count = Counter()
        self.labeling = None
        self.exact_match = False
        self.in_ring = True
        self.wild_card_vertices = []
        self.root_vertex = None
        self.root_group = defaultdict(list)
        # reorder vertices based on a construction path
        path = find_construction_path(graph, labeling_func=labeling_func)
        self.graph.reorder_vertices(path)
        # count atom usage
        for v in graph.vertices:
            if v.atom == ChemVertex.wild_card_atom:
                self.exact_match = True
                self.wild_card_vertices.append(v)
            else:
                self.atom_count[v.atom] += 1
                # check root vertex
                if v.root > 0:
                    if v.root == 1:
                        self.root_vertex = v
                    self.root_group[v.root].append(v.index)
        # set canonical labeling of the fragment
        if labeling_func is not None:
            self.labeling = labeling_func(self.graph)
        else:
            self.labeling = self.labeling_for_fragment()
        if logger.isEnabledFor(logging.INFO):
            logger.info('fragment:%s', self.to_string())

    def should_mark_vertex(self):
        """Check if matched path should be marked at vertices for incremental count.

        Returns:
            bool: true if vertices should be marked
        """
        return self.exact_match or len(self.root_group) > 0

    def saturated_root(self):
        """Check if bonds of root vertex is full

        Returns:
            bool: true if vertex is full
        """
        if self.root_vertex is not None:
            return len(self.root_group) == 1 and self.root_vertex.num_free_hand() < 1
        else:
            return False

    def labeling_for_fragment(self):
        """labeling graph of fragment.

        Returns:
            ChemGraphLabeling: labeling
        """
        # if fragement is rooted sub-structure, symmetry should be considered
        # with depth from root vertex
        original_symbol = dict()
        for v in self.graph.vertices:
            if v.root > 0:
                original_symbol[v] = v.color()
                v.set_color(v.color() + format(v.root))
        labeling = ChemGraphLabeling(self.graph.vertices)
        for v, symbol in original_symbol.items():
            v.set_color(symbol)
        return labeling

    def count_fragment_graph(self, graph, labeling=None):
        """Count the occurrence of the fragment in a graph.

        Args:
            graph (AtomGraph): a graph
            labeling (ChemGraphLabeling, optional): labeling of a graph. Defaults to None.

        Returns:
            int: the number of the occurrences
        """
        count, path = self.count_fragment_graph_with_path(graph, labeling)
        return count

    def count_fragment_graph_with_path(self, graph, labeling=None):
        """Count the occurrence of the fragment in a graph, and their paths.

        Args:
            graph (AtomGraph): a graph
            labeling (ChemGraphLabeling, optional): labeling of a graph. Defaults to None.

        Returns:
            int, list: the number of the occurrences, paths of matched fragments
        """
        # set labeling of a graph after reordering graph vertices if labeling is not given
        if labeling is None:
            graph = copy.deepcopy(graph)
            # path = find_construction_path(graph)
            # graph.reorder_vertices(path)
            graph.reorder_canonical()
            labeling = ChemGraphLabeling(graph.vertices)
        # make min_orbit map of vertex index of a graph
        if logger.isEnabledFor(logging.INFO):
            logger.info('count_fragment:%s auto:%s', graph.to_string(), labeling.automorphism.to_string())
        orbit_map = [labeling.automorphism.min_orbit(v.index) for v in graph.vertices]
        # count atom of graph
        atom_count = Counter()
        for v in graph.vertices:
            atom_count[v.atom] += 1
        # count fragments
        count = 0
        path = []
        if self.enough_atom(atom_count):
            if logger.isEnabledFor(logging.INFO):
                logger.info('check_fragment:%s', self.to_string())
            for v in graph.vertices:
                if v.index+1 >= self.graph.num_atom():
                    count0, path0 = self.count_fragment(v, orbit_map)
                    count += count0
                    path.extend(path0)
            if logger.isEnabledFor(logging.INFO):
                logger.info('check_fragment:%d', count)
        return count, path

    def count_fragment(self, new_v, orbit_map):
        """Count the new occurrence of the fragment after a new vertex is added to a graph.

        Args:
            new_v (AtomVertex): a vertex newly added to the graph
            orbit_map (list): a list of min index in the orbit of a vertex index

        Return:
            int, list: the number of occurrence, and a list of matched vertices in an exact match case
        """
        # make a map of min orbit for vertex
        forbit_map = [self.labeling.automorphism.max_orbit(fv.index) for fv in self.graph.vertices]
        total_count = 0
        fragment_path = []
        fv_candidate = [fv for fv in self.graph.vertices if fv.index == forbit_map[fv.index]]
        if len([e for e in new_v.edges if e.end.index < new_v.index]) <= 1:
            # new_v is a leaf of tree
            fv_candidate = [fv for fv in fv_candidate if len(fv.edges) <= 1]
        for fv in fv_candidate:
            f = fv
            v = new_v
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('count fragment:f=%d,v=%d %s', f.index, v.index, self.to_string())
            count, path, ring, visit, natom = self.search_subgraph(f, forbit_map, v, orbit_map, 0, v.index)
            if logger.isEnabledFor(logging.DEBUG):
                for p in path:
                    logger.debug('count_fragment:count=%d path=%s',
                                 count, [(f0.index, v0.index) for (f0, v0) in p])
            fragment_path.extend(path)
            total_count += count
        if total_count > 0:
            if logger.isEnabledFor(logging.INFO):
                logger.info('found fragment:v=%d count=%d fragment=%s', new_v.index, total_count, self.to_string())
        return total_count, fragment_path

    def count_fragment_edge(self, pos_v, new_v, orbit_map):
        """Count the new occurrence of the fragment include a new edge of a graph

        Args:
            pos_v (AtomVertex): a newly connected vertex
            new_v (AtomVertex): a connecting vertex to the graph
            orbit_map (list): a list of min index in the orbit of a vertex index

        Return:
            int, list: the number of occurrence, and a list of matched vertices in an exact match case
        """
        # get (new_v, pos_v) edge
        new_v_edge = None
        new_v_edge_index = 0
        for index, e in enumerate(new_v.edges):
            if e.end == pos_v:
                new_v_edge = e
                new_v_edge_index = index
                break
        if new_v_edge is None:
            logger.error('inconsistent edge')
            return 0, []
        # make a map of min orbit for vertex
        forbit_map = [self.labeling.automorphism.max_orbit(fv.index) for fv in self.graph.vertices]
        total_count = 0
        fragment_path = []
        fv_candidate = [fv for fv in self.graph.vertices if fv.index == forbit_map[fv.index]]
        if len(new_v.edges) <= 1:
            # new_v is a leaf of tree
            fv_candidate = [fv for fv in fv_candidate if len(fv.edges) <= 1]
        for fv in fv_candidate:
            f = fv
            v = new_v
            # check if new_v_edge matches one of edge from f
            match_edge_indices = set()
            for index, e in enumerate(f.edges):
                if e.bond_type == new_v_edge.bond_type and \
                        (e.start.atom == ChemVertex.wild_card_atom or
                         e.start.match_atom(new_v_edge.start)) and \
                        (e.end.atom == ChemVertex.wild_card_atom or
                         e.end.match_atom(new_v_edge.end)):
                    match_edge_indices.add(index)
            if len(match_edge_indices) == 0:
                continue
            v_index = len(orbit_map) - 1
            mandatory_match = (new_v_edge_index, match_edge_indices)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('count_fragment:f=%d,v=%d %s', f.index, v.index, self.to_string())
            count, path, ring, visit, natom = \
                self.search_subgraph(f, forbit_map, v, orbit_map, 0, v_index,
                                     mandatory_match=mandatory_match)
            if logger.isEnabledFor(logging.DEBUG):
                for p in path:
                    logger.debug('count_fragment:count=%d path=%s',
                                 len(path), [(f0.index, v0.index) for (f0, v0) in p])
            fragment_path.extend(path)
            total_count += count
        if total_count > 0:
            if logger.isEnabledFor(logging.INFO):
                logger.info('found fragment:v=%d count=%d fragment=%s', new_v.index, total_count, self.to_string())
        return total_count, fragment_path

    def mark_vertex(self, paths):
        """Mark vertices on the matched path for canceling the counter

        Args:
            paths (list): a list of matched paths

        Returns:
            list: a list of pairs of fragment and marked path
        """

        if self.exact_match:
            mark_path = self.extract_atom_path(paths)
        elif len(self.root_group) > 0:
            mark_path = self.extract_root_path(paths)
        else:
            return []
        fragment_path = []
        for path in mark_path:
            fragment_path.append((self, path))
            for vv in path:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('mark count=%d add fpath:v=%d fragment=%s, path=%s',
                                len(paths), vv.index, self.to_string(), [v0.index for v0 in path])
                vv.exact_match_fragment.add((self, path))
        return fragment_path

    def match_fragment_graph(self, graph, labeling=None):
        """Match the fragment to a graph.

        Args:
            graph (AtomGraph): a graph
            labeling (ChemGraphLabeling, optional): labeling of a graph. Defaults to None.

        Returns:
            list: a list of matched vertices pair
        """
        # set labeling of a graph after reordering graph vertices if labeling is not given
        vmap = dict()
        if labeling is None:
            orig_graph = graph
            graph = copy.deepcopy(graph)
            for index in range(len(orig_graph.vertices)):
                vmap[graph.vertices[index]] = orig_graph.vertices[index]
            graph.reorder_canonical()
            labeling = ChemGraphLabeling(graph.vertices)
        # make min_orbit map of vertex index of a graph
        if logger.isEnabledFor(logging.INFO):
            logger.info('match_fragment_graph:%s auto:%s', graph.to_string(), labeling.automorphism.to_string())
        orbit_map = [labeling.automorphism.min_orbit(v.index) for v in graph.vertices]
        # count atom of graph
        atom_count = Counter()
        for v in graph.vertices:
            atom_count[v.atom] += 1
        # get matching of fragments
        path = []
        if self.enough_atom(atom_count):
            if logger.isEnabledFor(logging.INFO):
                logger.info('match_fragment_graph:%s', self.to_string())
            for v in graph.vertices:
                if v.index+1 >= self.graph.num_atom():
                    path0 = self.match_fragment(v, orbit_map)
                    path.extend(path0)
            if logger.isEnabledFor(logging.INFO):
                for p in path:
                    logger.info('match_fragment_graph:path %s', [(f0.index, v0.index) for (f0, v0) in p])
            if len(vmap) > 0:
                new_path = []
                for path0 in path:
                    new_path.append([(f, vmap[v]) for f, v in path0])
                path = new_path
            if logger.isEnabledFor(logging.INFO):
                for p in path:
                    logger.info('match_fragment_orig_graph:path %s', [(f0.index, v0.index) for (f0, v0) in p])
        return path

    def match_fragment_graph_vertex(self, fvertex, vertex, graph, labeling):
        """Match the fragment to a graph including a vertex.

        Args:
            fvertex (AtomVertex): a vertex of a fragment
            vertex (AtomVertex): a vertex of a graph
            graph (AtomGraph): a graph
            labeling (ChemGraphLabeling): labeling of a graph.

        Returns:
            list: a list of matched vertices pair
        """
        if vertex is None:
            return self.match_fragment_graph(graph, labeling=labeling)

        if logger.isEnabledFor(logging.INFO):
            logger.info('match_fragment_graph:fvertex=%d %s', fvertex.index, self.graph.to_string())
            logger.info('match_fragment_graph:vertex=%d %s auto:%s', vertex.index, graph.to_string(),
                        labeling.automorphism.to_string())
        # make min_orbit map of vertex index of a graph
        orbit_map = [labeling.automorphism.min_orbit(v.index) for v in graph.vertices]
        # make max_orbit map of vertex index of a fragment graph
        forbit_map = [self.labeling.automorphism.max_orbit(fv.index) for fv in self.graph.vertices]
        # matching a fragment to a graph starting with (fvertec, vertex)
        count, path, ring, visit, natom = \
            self.search_subgraph(fvertex, forbit_map, vertex, orbit_map, 0, len(graph.vertices))
        if logger.isEnabledFor(logging.INFO):
            for p in path:
                logger.info('match_fragment_graph:path %s', [(f0.index, v0.index) for (f0, v0) in p])
        return path

    def match_fragment(self, new_v, orbit_map):
        """Match the fragment and get a list of matching vertex pairs.

        Args:
            new_v (AtomVertex): a vertex newly added to the graph
            orbit_map (list): a list of min index in the orbit of a vertex index

        Return:
            list: a list of matched vertices in an exact match
        """
        # make a map of min orbit for vertex
        forbit_map = [self.labeling.automorphism.max_orbit(fv.index) for fv in self.graph.vertices]
        fragment_path = []
        fv_candidate = [fv for fv in self.graph.vertices if fv.index == forbit_map[fv.index]]
        if len([e for e in new_v.edges if e.end.index < new_v.index]) <= 1:
            # new_v is a leaf of tree
            fv_candidate = [fv for fv in fv_candidate if len(fv.edges) <= 1]
        else:
            # new_v is a connection of a ring
            pass
            # fv_candidate = [fv for fv in fv_candidate if len(fv.edges)==1 or fv.ring_atom()]
        for fv in fv_candidate:
            f = fv
            v = new_v
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('count_fragment:f=%d,v=%d %s', f.index, v.index, self.to_string())
            count, path, ring, visit, natom = self.search_subgraph(f, forbit_map, v, orbit_map, 0, v.index)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('count_fragment:count=%d', count)
            fragment_path.extend(path)
        return fragment_path

    def search_subgraph(self, f, orbit_map_f, v, orbit_map_v, depth, max_index, reach_count=False,
                        mandatory_match=None):
        """Matching the fragment to a graph by depth first search.

        Args:
            f (AtomVertex): a vertex of a fragment
            orbit_map_f (list): a list of min index in the orbit of a vertex index of a fragment
            orbit_map_v (list): a list of min vertices  in the orbit of a vertex index of a fragment
            v (AtomVertex): a vertex of a graph
            orbit_map_f (list): a list of max index in the orbit of a vertex index of a graph
            depth (int): current depth of depth first search
            max_index (int): maximum index of a vertex which should be followed
            reach_count (bool, optional): flag of counting maximum partial matching. Defaults to False.
            mandatory_match (tuple, optional): a pair of v index and f indices which should match. Defaults to None.

        Returns:
            int, list, bool, int, int: the number of occurrence, a list of matched vertices,
                flag of ring vertices, depth of closed ring, max partial matching
        """
        if f.atom == ChemVertex.wild_card_atom:
            if len(f.edges) > len(v.edges):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('degree mismatch:f=%d,v=%d', f.index, v.index)
                return 0, [], False, 0, 0
        else:
            if not f.match_atom(v):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('atom mismatch:f=%d,v=%d', f.index, v.index)
                return 0, [], False, 0, 0
            if self.exact_match:
                if len(f.edges) != len(v.edges):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('exact match degree mismatch:f=%d,v=%d', f.index, v.index)
                    return 0, [], False, 0, 0 if len(f.edges) < len(v.edges) else 1
            elif f.root > 0:
                if len(f.edges) != len(v.edges):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('root atom degree mismatch:f=%d,v=%d', f.index, v.index)
                    return 0, [], False, 0, 0 if len(f.edges) < len(v.edges) else 1

            if len(f.edges) > len(v.edges):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('degree mismatch:f=%d,v=%d', f.index, v.index)
                return 0, [], False, 0, 1
            if not self.in_ring:
                if (not f.in_ring) and v.in_ring:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('in-ring mismatch:f=%d,v=%d', f.index, v.index)
                    return 0, [], False, 0, 0
        if f.visit != v.visit:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('ring mismatch:f=%d,v=%d,fring=%d,vring=%d', f.index, v.index, f.visit, v.visit)
            return 0, [], False, 0, 0
        if f.visit > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('match ring:f=%d,v=%d,ring=%d', f.index, v.index, f.visit)
            # ring end of fragment
            return 1, [[]], v.ring_atom(), f.visit, 1
        if (depth > 0 and len(f.edges) == 1) or (depth == 0 and len(f.edges) == 0):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('match:f=%d,v=%d', f.index, v.index)
            # leaf of fragment
            return 1, [[(f, v)]], v.ring_atom(), depth+1, 1

        # set depth to vertex
        f.visit = depth+1
        v.visit = depth+1

        # initialize variables
        vvlist = [ve for ve in v.edges if (depth == 0 or ve.end.visit != depth) and ve.end.index <= max_index]
        fvlist = [fe for fe in f.edges if depth == 0 or fe.end.visit != depth]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('scan f=%d v=%d fvlist=%s vvlist=%s depth=%d',
                         f.index, v.index,
                         [fe.end.index for fe in fvlist], [ve.end.index for ve in vvlist], depth)
        vindex = [-1]*len(fvlist)  # current vindex for fvlist[findex0]
        findex = [-1]*len(vvlist)  # current findex for vvlist[vindex0] -1 if empty
        vindex_orbit = [[-1] for _ in orbit_map_f]
        findex_orbit = [[] for _ in orbit_map_v]
        match_path = [[] for _ in fvlist]
        match_ring = [False]*len(fvlist)
        match_visit = [0]*len(fvlist)
        fcount_total = 0
        fragment_path = []
        has_ring = False
        visit_ring = f.visit+1
        max_reach = 0

        # try all the pair of <fv,vv> for fv in fvlist, vv in vvlist
        # however,
        # for the fv1, fv2 of the same orbit of fvlist, fv1 < fv2 -> vv1 < vv2
        # additionally, for the vv1, vv2 of the same orbit of vvlist, vv1 < vv2 -> fv1 < fv2

        # above restriction may not work for a graph with cycles.
        # for symmetry vertices of a fragment, matching is restricted.
        # {f1,f2} in a orbit, (f1,v1) (f2,v2) is enough, (f1,v2) (f2,v1) should not be counted
        # for symmetry vertices of a target graph, matching is also restricted.
        # {v1,v2} in a orbit, (f1,v1) (f2,v2) is enough, (f2,v1) (f1,v2) should not be counted
        # if there is a cycle in a graph, matching order to symmetry vertices {f1,f2}
        # cannot be predetermined, other symmetry vertices {f3,f4} may be inter-related to {f1,f2}

        visit_atoms = [0] * len(fvlist)
        findex0 = 0
        while findex0 >= 0:
            # find empty position of vindex
            findex0_orbit = orbit_map_f[fvlist[findex0].end.index]
            # vindex of the same orbit findex must be increasing
            next_vindex = max(vindex[findex0]+1, vindex_orbit[findex0_orbit][-1]+1)
            vindex0 = -1
            for vind in range(next_vindex, len(findex)):
                if findex[vind] < 0:
                    # check edge bound
                    if not fvlist[findex0].match_bond(vvlist[vind]):
                        if logger.isEnabledFor(logging.DEBUG):
                            fe = fvlist[findex0]
                            ve = vvlist[vind]
                            logger.debug('bond mismatch:f=%d(%d-%d),v=%d(%d-%d)',
                                         findex0, fe.start.index, fe.end.index,
                                         vind, ve.start.index, ve.end.index)
                        continue
                    # findex of the same obrit vindex must increasing, if findex in the same orbit
                    vorbit_consistent = True
                    for find in findex_orbit[orbit_map_v[vvlist[vind].end.index]]:
                        if vindex[find] > vind and orbit_map_f[fvlist[find].end.index] == findex_orbit:
                            vorbit_consistent = False
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug('vorbit inconsistent:f=%d,v=%d', findex0, vind)
                            break
                    if not vorbit_consistent:
                        continue
                    # check mandatory edge match
                    if mandatory_match is not None:
                        (mandatory_v_index, mandatory_f_indices) = mandatory_match
                        if vind == mandatory_v_index:
                            if findex0 not in mandatory_f_indices:
                                continue
                        elif findex[mandatory_v_index] not in mandatory_f_indices:
                            filled_count = 0
                            for find0 in mandatory_f_indices:
                                if vindex[find0] >= 0 or find0 == findex0:
                                    filled_count += 1
                            if filled_count == len(mandatory_f_indices):
                                continue

                    # vind is not used
                    findex[vind] = findex0
                    findex_orbit[orbit_map_v[vvlist[vind].end.index]].append(findex0)
                    vindex[findex0] = vind
                    vindex_orbit[findex0_orbit].append(vind)
                    vindex0 = vind
                    break

            reset_findex = False
            if vindex0 >= 0:
                # count matching for f=findex0 v=vindex0
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('try findex0=%d(%d) vindex0=%d(%d) depath=%d',
                                 findex0, fvlist[findex0].end.index, vindex0, vvlist[vindex0].end.index, depth)
                # check loop scan
                fring_loop_reach = False
                vring_loop_reach = False
                if f.ring_atom():
                    for idx in range(findex0):
                        for path in match_path[idx]:
                            for (f0, v0) in path:
                                if f0 == fvlist[findex0].end:
                                    fring_loop_reach = True
                                    if v0 == vvlist[vindex0].end:
                                        vring_loop_reach = True
                                    break
                        if fring_loop_reach:
                            break
                if fring_loop_reach:
                    # fvlist[findex0] is already scanned in reverse direction
                    if vring_loop_reach:
                        # match vvlist[vindex0]
                        count = 1
                        path = [[]]
                        ring = True
                        visit = depth+1
                        visit_atoms[findex0] = 1
                        # modify vindex_orbit
                        # erase ordering restriction by current findex0
                        if orbit_map_f[fvlist[findex0].end.index] != fvlist[findex0].end.index:
                            vindex_orbit[findex0_orbit][-1] = vindex_orbit[findex0_orbit][-2]
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('ring loop ok:f=%d,v=%d depath=%d', findex0, vindex0, depth)
                    else:
                        # mismatch vvlist[vindex0]
                        count = 0
                        path = []
                        ring = False
                        visit = f.visit+1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('ring loop ng:f=%d,v=%d depath=%d', findex0, vindex0, depth)
                else:
                    # start scan from fvlist[findex0] and vvlist[vindex0]
                    count, path, ring, visit, natom = \
                        self.search_subgraph(fvlist[findex0].end, orbit_map_f,
                                             vvlist[vindex0].end, orbit_map_v,
                                             depth+1, max_index, reach_count=reach_count)
                    visit_atoms[findex0] = natom
                    if visit <= depth+1 and \
                            orbit_map_f[fvlist[findex0].end.index] != fvlist[findex0].end.index:
                        # check inclusion of symmetry vertices
                        for p in path:
                            symmetry_count = 0
                            for (f0, v0) in p:
                                if orbit_map_f[f0.index] != f0.index:
                                    symmetry_count += 1
                            symmetry_count += max(0, 2-len(p))
                            if symmetry_count > 1:
                                # modify vindex_orbit
                                # erase ordering restriction by current findex0
                                vindex_orbit[findex0_orbit][-1] = vindex_orbit[findex0_orbit][-2]
                            break

                if count > 0:
                    # found matching
                    match_path[findex0] = path
                    match_ring[findex0] = ring
                    match_visit[findex0] = visit
                    if findex0 == len(fvlist)-1:
                        # all the findex is scanned
                        has_ring = (has_ring or any(match_ring))
                        visit_ring = min(visit_ring, min(match_visit))
                        merged_path = self.merge_match_path(match_path, any(match_ring))
                        if min(match_visit) <= depth+1:
                            # check duplication path
                            merged_path = self.check_duplication_path(merged_path, fragment_path, orbit_map_f)
                        count_total = len(merged_path)
                        fcount_total += count_total
                        fragment_path.extend(merged_path)
                        if logger.isEnabledFor(logging.DEBUG):
                            for mpath in merged_path:
                                logger.debug('scan all %d path=%s depth=%d visit=%d',
                                             count_total, [(ff.index, vv.index) for (ff, vv) in mpath],
                                             depth, visit_ring)
                        # reset findex0 to the position of remaining trial
                        reset_findex = True
                    else:
                        # proceed to next findex0
                        findex0 += 1
                else:
                    # fail matching
                    # reset findex to the position of remaining trial
                    reset_findex = True
            else:
                # all the vvlist is tried
                if reach_count:
                    # update max_reach
                    max_reach = max(max_reach, self.count_max_reach(f, v, fvlist, vvlist, visit_atoms))
                # reset visit atoms
                visit_atoms[findex0] = 0
                # reset findex
                vindex[findex0] = -1
                findex0 -= 1
                findex0_orbit = orbit_map_f[fvlist[findex0].end.index]
                # reset findex to the position of remaining trial
                reset_findex = True

            if reset_findex:
                if findex0 >= 0 and reach_count:
                    # update max_reach
                    max_reach = max(max_reach, self.count_max_reach(f, v, fvlist, vvlist, visit_atoms))
                while findex0 >= 0:
                    findex[vindex[findex0]] = -1
                    findex_orbit[orbit_map_v[vvlist[vindex[findex0]].end.index]].pop()
                    vindex_orbit[findex0_orbit].pop()
                    if vindex[findex0] == len(vvlist)-1:
                        # end of vvlist, reset
                        vindex[findex0] = -1
                    else:
                        break
                    # reset visit atoms
                    visit_atoms[findex0] = 0
                    # reset findex
                    vindex[findex0] = -1
                    findex0 -= 1
                    findex0_orbit = orbit_map_f[fvlist[findex0].end.index]

        # reset depth of vertex
        f.visit = 0
        v.visit = 0
        # append vertex to fragment_path
        for path in fragment_path:
            path.append((f, v))
        if logger.isEnabledFor(logging.DEBUG):
            if len(fragment_path) > 0:
                for fpath in fragment_path:
                    logger.debug('scan end %d path=%s depth=%d visit=%d',
                                 fcount_total, [(ff.index, vv.index) for (ff, vv) in fpath], depth, visit_ring)
            else:
                logger.debug('scan end %d path=[] depth=%d visit=%d', fcount_total, depth, visit_ring)
        return fcount_total, fragment_path, has_ring, visit_ring, max_reach

    @staticmethod
    def count_max_reach(f, v, fvlist, vvlist, visit_atoms):
        """Count the partial matching atoms

        Args:
            f (AtomVertex): vertex of fragment
            v (AtomVertex): vertex of target graph
            fvlist (list): a list of edges of fragment
            vvlist (list): a list of edges of target graph
            visit_atoms (list): number of matched atoms for each edges

        Returns:
            int: the number of partial matching atoms
        """
        max_reach = 0
        total_reach = 0
        success_edge = 0
        failed_bond = 0
        for r_ind, reach in enumerate(visit_atoms):
            total_reach += reach
            if reach > 0:
                # success
                success_edge += 1
            else:
                # failure
                if f.ring_atom() and fvlist[r_ind].end.ring_atom():
                    # not necessarily failure (count as success)
                    success_edge += 1
                else:
                    failed_bond += fvlist[r_ind].get_bond_order()
        if f.root > 0:
            if len(vvlist) == success_edge:
                max_reach = total_reach + 1
        else:
            if v.num_free_hand() >= failed_bond:
                max_reach = total_reach + 1
        return max_reach

    @staticmethod
    def extract_atom_path(fragment_path):
        """Extract vertex path excluding a vertex of wind card atom.

        Args:
            fragment_path (list): a list of a list of search path of a pair of fragment vertex and graph vertex

        Returns:
            list: a list of a list of search path excluding a vertex of wild card atom
        """
        atom_path = []
        for path in fragment_path:
            new_path = []
            for (f, v) in path:
                if f.atom != ChemVertex.wild_card_atom:
                    new_path.append(v)
            atom_path.append(tuple(new_path))
        return atom_path

    @staticmethod
    def extract_root_path(fragment_path):
        """Extract single vertex path for each root vertex.

        Args:
            fragment_path (list): a list of a list of search path of a pair of fragment vertex and graph vertex

        Returns:
            list: a list of a list of search path including unique root vertex
        """
        root_path = []
        root_set = set()
        for path in fragment_path:
            new_path = []
            root = None
            for (f, v) in path:
                if f.root > 1:
                    new_path.append(v)
                elif f.root == 1:
                    root = v
            if root is None:
                # no root vertex
                roots = tuple(sorted([v0.index for v0 in new_path]))
                if roots not in root_set:
                    root_path.append(tuple(new_path))
                    root_set.add(roots)
            else:
                # has root vertex (fp structure)
                if root not in root_set:
                    new_path.append(root)
                    root_path.append(tuple(new_path))
                    root_set.add(root)
        return root_path

    @staticmethod
    def merge_match_path(match_path, has_ring):
        """Merge matching search path of vertices like a Cartesian product. If target graph include rings
        overlap of the paths are checked and excluded from merging process.

        Args:
             match_path (list): a list of a list of search path starting from each connecting edges
             has_ring (bool): a flag of rings in search path

        Returns:
            list: a list of a list of search path
        """
        merged_path = []
        ring_vertices = []
        for path in match_path:
            if len(path) == 0:
                continue
            elif len(merged_path) == 0:
                for path0 in path:
                    merged_path.append(path0)
                    # collect ring vertices
                    rv_list = []
                    if has_ring:
                        for (f, v) in path0:
                            if v.ring_atom():
                                rv_list.append(v)
                    ring_vertices.append(rv_list)
            else:
                new_merged_path = []
                new_ring_vertices = []
                for mpath, rvertices in zip(merged_path, ring_vertices):
                    for path0 in path:
                        # check duplication ring vertices
                        duplication = False
                        rv_list = []
                        if has_ring:
                            for (f, v) in path0:
                                if v.ring_atom():
                                    if v in rvertices:
                                        duplication = True
                                        break
                                    else:
                                        rv_list.append(v)
                        if not duplication:
                            new_merged_path.append(mpath+path0)
                            new_ring_vertices.append(rvertices+rv_list)
                merged_path = new_merged_path
                ring_vertices = new_ring_vertices
                if len(merged_path) == 0:
                    return []
        if logger.isEnabledFor(logging.DEBUG):
            for mpath in merged_path:
                logger.debug('merged path: %s', [(ff.index, vv.index) for (ff, vv) in mpath])
        return merged_path

    @staticmethod
    def check_duplication_path(merged_path, fragment_path, orbit_map_f):
        """Check the duplication of matching path

        Args:
            merged_path (list): a list of newly found matching path
            fragment_path (list): a list of existing matching path
            orbit_map_f (list): a map of an orbit of vertices index

        Returns:
            list: a list of unique matching paths
        """
        merged_path0 = list()
        for path0 in merged_path:
            path0 = [(orbit_map_f[pf.index], pv.index) for (pf, pv) in path0]
            path0 = sorted(path0, key=itemgetter(0, 1))
            merged_path0.append(path0)
        fragment_path0 = list()
        for path0 in fragment_path:
            path0 = [(orbit_map_f[pf.index], pv.index) for (pf, pv) in path0]
            path0 = sorted(path0, key=itemgetter(0, 1))
            fragment_path0.append(path0)
        checked_path = []
        for path0, pathm in zip(merged_path0, merged_path):
            same_path = False
            for pathf in fragment_path0:
                same_path0 = True
                for (p0, p1) in zip(path0, pathf):
                    if p0[0] != p1[0] or p0[1] != p1[1]:
                        same_path0 = False
                        break
                if same_path0:
                    same_path = True
                    break
            if not same_path:
                checked_path.append(pathm)
        return checked_path

    def enough_atom(self, graph_atom_count):
        """Check if a graph has enough vertices to check the occurrence of a fragment.

        Args:
            graph_atom_count (Counter): counter of atoms in a graph

        Returns:
            bool: True is graph is large enough, False otherwise
        """
        for atom, usage in self.atom_count.items():
            if atom not in graph_atom_count:
                return False
            if graph_atom_count[atom] < usage:
                return False
        if len(self.wild_card_vertices) > 0:
            atom_count = sum(graph_atom_count.values())
            if atom_count < len(self.graph.vertices):
                return False
        return True

    def to_string(self):
        """Get a string representation of a fragment
        Returns:
            str: a string representation
        """
        root_string = ''
        if len(self.root_group) > 0:
            root_string = 'r'
            for r in sorted(self.root_group.keys()):
                indices = self.root_group[r]
                root_string += '['
                for index in indices:
                    if index == indices[-1]:
                        root_string += '%d]' % index
                    else:
                        root_string += '%d,' % index
            root_string += ' '
        return '%s%s auto:%s' % (root_string, self.graph.to_string(), self.labeling.automorphism.to_string())

    def to_string_label(self):
        """Get a string representation of a fragment by labeled index
        Returns:
            str: a string representation
        """
        root_string = ''
        if self.root_vertex:
            root_string = 'r'
            for r in sorted(self.root_group.keys()):
                indices = self.labeling.permutation.apply_list(self.root_group[r])
                root_string += '['
                for index in indices:
                    if index == indices[-1]:
                        root_string += '%d]' % index
                    else:
                        root_string += '%d,' % index
            root_string += ' '
        autom = copy.deepcopy(self.labeling.automorphism)
        self.labeling.permutation.apply_autom(autom)
        return '%s%s autom:%s' % (root_string, self.graph.to_string_label(), autom.to_string())

    def __repr__(self):
        return self.to_string()
