# -*- coding:utf-8 -*-
"""
ChemRingGen.py

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
from .ChemGraphFragment import *
from .ChemGraphUtil import NumRange

import copy

import logging
logger = logging.getLogger(__name__)


class ChemRingGenerator(object):
    """Generation of pattern of atoms in a ring
    by replacing carbons (by default) with specified number of other atoms.

    Attributes:
         graph (AtomGraph): a molecular graph replacing carbons
         labeling (ChemGraphLabeling): labeling of a graph
         atom_resource (dict): a dictionary of replacing atoms
         target_atom (str): target atom name for replacing
    """

    default_target_atom = 'C'
    """str: default target atom"""

    def __init__(self, graph):
        """Constructor of ring generator.

        Args:
            graph (AtomGraph): a graph of a ring
        """
        self.fragment = ChemFragment(copy.deepcopy(graph))
        self.graph = copy.deepcopy(self.fragment.graph)
        self.labeling = ChemGraphLabeling(self.graph.vertices)
        self.atom_resource = {}
        self.target_atom = None

    def reorder_vertex_for_replacing_atom(self, graph, replace_atom):
        """Reorder vertex indices so that non-target atoms have smaller indices.

        Args:
            graph (AtomGraph): a graph of a ring
            replace_atom (set): a set of replacing atom name

        Returns:
            Automorphism: an automorphism of a graph to replace
       """
        fix_vmap = {}
        ignore_vmap = {}
        other_vmap = {}
        orig_vmap = {}
        reorder_index0 = [0] * len(graph.vertices)
        reorder_index1 = [0] * len(graph.vertices)
        reorder_index = [0] * len(graph.vertices)
        for v in graph.vertices:
            if v.atom in replace_atom:
                if v.num_free_hand() < 0 or not v.ring_atom():
                    fix_vmap[v] = v.color()
                else:
                    ignore_vmap[v] = v.color()
                    v.set_color(self.target_atom)
            elif v.atom != self.target_atom:
                other_vmap[v] = v.color()
                v.set_color(' '+v.color())
            else:
                orig_vmap[v] = v.color()
        vertex_list = list(other_vmap.keys())
        vertex_list.extend(list(orig_vmap.keys()))
        vertex_list.extend(list(ignore_vmap.keys()))
        vertex_list.extend(list(fix_vmap.keys()))
        for index, v in enumerate(vertex_list):
            reorder_index0[v.index] = index
        graph.reorder_vertices(vertex_list)
        ChemGraphLabeling(graph.vertices)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('reorder before:%s', graph.to_string())
        vertex_list = sorted(vertex_list, key=lambda x: x.label)
        for index, v in enumerate(vertex_list):
            reorder_index1[v.index] = index
        graph.reorder_vertices(vertex_list)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('reorder after:%s', graph.to_string())
        for v, symbol in other_vmap.items():
            v.set_color(symbol)
        labeling = ChemGraphLabeling(graph.vertices)
        for v, symbol in ignore_vmap.items():
            v.set_color(symbol)
        for index in range(len(vertex_list)):
            reorder_index[index] = reorder_index1[reorder_index0[index]]
        return labeling.automorphism, reorder_index

    def generate_ring_graphs(self, atom_resource, target_atom=None):
        """Generate rings of different atom patterns.

        Args:
            atom_resource (dict): dictionary of available atoms for replacing
            target_atom (str, optional): target atom name for replacing. Defaults to None.

        Returns:
            list: a list of ring graphs
        """
        logger.info('generate rings: {0} {1} replace={2}'.format(self.graph.to_smiles(), self.graph.to_string(),
                                                                 atom_resource))
        # check graph size
        if self.graph.num_atom() == 0:
            return []

        # identify replace target
        if target_atom is not None:
            self.target_atom = target_atom
        else:
            target_set = set()
            for v in self.graph.vertices:
                if v.atom not in atom_resource:
                    target_set.add(v.atom)
            if len(target_set) > 1:
                if self.default_target_atom in target_set:
                    logger.info('replace target is set to default C')
                    self.target_atom = self.default_target_atom
                else:
                    logger.error('replace target is not unique')
                    return []
            elif len(target_set) > 0:
                self.target_atom = target_set.pop()
            else:
                logger.info('no replace target')
                return [copy.deepcopy(self.graph)]

        self.atom_resource = {}
        atom_count = Counter()
        for atom, nrange in atom_resource.items():
            atom_range = NumRange(nrange)
            if atom_range.max > 0:
                self.atom_resource[atom] = NumRange(nrange)
                atom_count[atom] = atom_range.max

        if len(atom_count) == 0:
            return [copy.deepcopy(self.graph)]

        # check satisfaction of original graph
        new_graphs = []
        include_this = True
        for atom, nrange in self.atom_resource.items():
            if not nrange.contains(0):
                include_this = False
        if include_this:
            new_graphs = [copy.deepcopy(self.graph)]

        # replace atom with target atom if included in replacing atoms
        initial_graph = copy.deepcopy(self.graph)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('initial graph: {0} {1}'.format(initial_graph.to_smiles(), initial_graph.to_string()))
        replace_map = dict()
        replace_atom_count = Counter()
        fix_atom_count = Counter()
        for v in initial_graph.vertices:
            if (v.num_free_hand() < 0 or not v.ring_atom()) and v.atom != self.target_atom:
                fix_atom_count[v.atom] += 1
        for v in initial_graph.vertices:
            if v.atom in self.atom_resource:
                if v.atom not in fix_atom_count.keys():
                    replace_map[v.index] = copy.copy(v)
                    replace_atom_count[v.atom] += 1
                    v.replace_symbol(self.target_atom)
                    v.explicit_h = 0
                else:
                    if v.num_free_hand() < 0 or not v.ring_atom():
                        atom_count[v.atom] -= 1
        initial_labeling = ChemGraphLabeling(initial_graph.vertices)

        # generate rings by replacing atoms
        new_graphs0 = self.generate_ring_graphs0(initial_graph, None, None, atom_count, None)

        result_graphs = []
        if len(replace_map):
            replace_key = list(replace_map.keys())
            orbits = initial_labeling.automorphism.orbit_list(replace_key)
            replace_maps = [replace_map]
            for orbit in orbits:
                new_replace_map = {}
                for index, orbit_index in enumerate(orbit):
                    new_replace_map[orbit_index] = replace_map[replace_key[index]]
                replace_maps.append(new_replace_map)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('orig graph: {0} {1}'.format(initial_graph.to_smiles(), initial_graph.to_string()))
                logger.debug('autom     : {0}'.format(initial_labeling.automorphism.to_string()))
                logger.debug('replace map: {0}'.format(replace_maps))
            for graph in new_graphs0:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('new graph: {0} {1}'.format(graph.to_smiles(), graph.to_string()))
                for replace in replace_maps:
                    match_replace = True
                    for index, vertex in replace.items():
                        if graph.vertices[index].atom != vertex.atom:
                            match_replace = False
                            break
                    if match_replace:
                        # restore explicit h from original vertex
                        for index, vertex in replace.items():
                            graph.vertices[index].explicit_h = vertex.explicit_h
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('matched: {0}'.format({index: v.atom for index, v in replace.items()}))
                        result_graphs.append(graph)
                        break
            return result_graphs
        else:
            result_graphs = new_graphs
            result_graphs.extend(new_graphs0)
            return new_graphs

    def generate_ring_graphs0(self, graph, labeling, autom, atom_count, prev_v):
        """Generate rings of different atom pattern by replacing target atom one by one.

        Args:
            graph (AtomGraph): current ring graph
            labeling (ChemGraphLabeling, None): labeling of a graph
            autom (Automorphism, None): automorphism of a graph
            atom_count (Counter): maximum number of available atoms for replacing
            prev_v: a vertex whose atom is replaced last

        Returns:
            list: a list of rings
        """
        new_graphs = []
        replace_atom = set([atom for atom, count in atom_count.items() if count > 0])
        reorder_index = None

        if prev_v is None:
            # start of new atom
            atom, count = atom_count.popitem()
            next_index = 0
            graph = copy.deepcopy(graph)
            autom, reorder_index = self.reorder_vertex_for_replacing_atom(graph, replace_atom)
            labeling = ChemGraphLabeling(graph.vertices)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('new atom=%s autom=%s graph=%s', atom, autom.to_string(), graph.to_string())
        else:
            # continue same atom
            atom = prev_v.atom
            count = atom_count[atom]
            atom_count.pop(atom)
            next_index = prev_v.index+1

        # replace for new atom symbol
        num_atom_range = self.atom_resource[atom]
        if num_atom_range.contains(num_atom_range.max-count) and len(atom_count) > 0:
            new_graphs0 = self.generate_ring_graphs0(graph, None, None, atom_count, None)
            new_graphs.extend(new_graphs0)

        # continue replacement for atom of previous vertex
        if count > 0:
            # get candidate vertex of replacement
            replace_candidate = []
            for vv in graph.vertices[next_index:]:
                if vv.atom == self.target_atom and vv.ring_atom() and labeling.is_min_orbit(vv.index):
                    replace_candidate.append(vv)

            for v in replace_candidate:
                original_symbol = v.atom
                atom_count[atom] = count - 1
                new_v = v.replace_symbol(atom)
                if new_v is not None:
                    if True:  # graph.num_all_free_hand() >= -1:
                        if self.min_graph(graph, autom, new_v):
                            labeling.labeling()
                            if num_atom_range.contains(num_atom_range.max-atom_count[atom]):
                                include_this = True
                                for atom0 in atom_count:
                                    if atom0 != atom and not self.atom_resource[atom0].contains(0):
                                        include_this = False
                                if include_this:
                                    new_graphs.append(copy.deepcopy(graph))
                            new_graphs0 = self.generate_ring_graphs0(graph, labeling, autom, atom_count, v)
                            new_graphs.extend(new_graphs0)
                    new_v.replace_symbol(original_symbol)
                atom_count[atom] = count

        # restore the vertex index
        if reorder_index is not None:
            for new_graph in new_graphs:
                new_vertex_list = [new_graph.vertices[reorder_index[v.index]] for v in new_graph.vertices]
                new_graph.reorder_vertices(new_vertex_list)

        return new_graphs

    def min_graph(self, graph, automorphism, new_vertex):
        """Check if a pattern of newly replaced atom are lexicographically minimum.

        Args:
            graph (AtomGraph): a graph
            automorphism (Automorphism): an automorphism of a graph
            new_vertex: a newly replaced vertex

        Returns:
            True if minimum in equivalent labeling. False otherwise.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('min_graph:v=%d autom=%s graph=%s',
                         new_vertex.index, automorphism.to_string(), graph.to_string())
        start_orbit = sorted([v.index for v in graph.vertices if v.atom == new_vertex.atom])
        min_orbit = start_orbit
        orbit_list = [start_orbit]
        new_orbit_list = [start_orbit]
        while len(new_orbit_list) > 0:
            old_new_orbit_list = new_orbit_list
            new_orbit_list = []
            for lst in old_new_orbit_list:
                new_orbit = automorphism.orbit_list(lst)
                for index_list in new_orbit:
                    index_list = sorted(index_list)
                    # check atom of orbit
                    atom_conflict = False
                    for i in index_list:
                        v = self.graph.vertices[i]
                        if v.atom != new_vertex.atom and v.atom in self.atom_resource:
                            atom_conflict = True
                            break
                    if not atom_conflict:
                        if self.lexcographic_order(index_list, min_orbit) < 0:
                            min_orbit = index_list
                    found_new_orbit = True
                    for ll in orbit_list:
                        if self.lexcographic_order(index_list, ll) == 0:
                            found_new_orbit = False
                            break
                    if found_new_orbit:
                        new_orbit_list.append(index_list)
                        orbit_list.append(index_list)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('new_orbit=%s orbit_list=%s', new_orbit_list, orbit_list)
        return self.lexcographic_order(min_orbit, start_orbit) == 0

    @staticmethod
    def lexcographic_order(list1, list2):
        """Compare lexcographic order of orbits.

        Args:
            list1 (list): a list of indices
            list2 (list): a list of indices

        Returns:
            int: -1 if list1 < list2, 0 if list1 == list2, 1 if list1 > list2
        """
        for l1, l2 in zip(list1, list2):
            if l1 < l2:
                return -1
            elif l1 > l2:
                return 1
        return 0
