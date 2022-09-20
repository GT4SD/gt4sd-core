# -*- coding:utf-8 -*-
"""
ChemGraphLabeling.py

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

from array import array
from collections import defaultdict, Counter

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ChemGraphLabeling(object):
    """Canonical graph labeling. Automorphism of a graph is also obtained.
    In the algorithm, vertices are assumed to be indexed starting from 0 and sequential.

    Attributes:
        vertices (list): a list of vertices of a graph
        zero_check (bool): checking zero label first
        permutation (array): labeling as a permutation
        automorphism (AutoMorphism): automorphism of a graph as a list of permutations
    """

    @staticmethod
    def possible_zero_check(vertices, last_vertex):
        pos = defaultdict(set)
        for v in sorted(vertices, key=lambda x: x.index, reverse=True):
            pos[(v.num_edge(), 0 if v.ring_atom() else 1, v.color(), v.num_bond2())].add(v.index)
        partition = [pos[key] for key in sorted(pos.keys())]
        return last_vertex.index in partition[0]

    def __init__(self, vertices, zero_check=False):
        """Constructor of graph labeling.

        Args:
            vertices (list): a list of vertices of a graph
            zero_check (bool, optional): checking zero label first. Defaults to False.
        """
        self.vertices = sorted(vertices, key=lambda x: x.index, reverse=True)
        self.zero_check = zero_check
        self.permutation = None
        self.automorphism = None
        if not self.zero_check:
            self.labeling()

    def labeling(self, zero_index=0):
        """Canonical labeling of vertices of a graph based on McKay's labeling algorithm.

        Args:
            zero_index (int): index of vertices for zero check

        Returns:
            bool: True if success in zero_check
        """
        # make an initial partition
        pos = defaultdict(list)
        for v in self.vertices:
            pos[(v.num_edge(), 0 if v.ring_atom() else 1, v.color(), v.num_bond2())].append(v)
        partition = [pos[key] for key in sorted(pos.keys())]
        if logger.isEnabledFor(logging.INFO):
            logger.info('root:%s', self.print_partition(partition, []))

        if self.zero_check and len(partition) > 0:
            if zero_index not in [v.index for v in partition[0]]:
                return False

        # search for canonical labeling
        check_list = [True]*len(partition)
        split = []
        permutation = []
        automorphism = AutoMorphism()
        if not self.labeling0(partition, check_list, 0, split, permutation, automorphism, zero_index):
            return False

        # save labeling result
        self.permutation = permutation[0]
        self.automorphism = automorphism
        if logger.isEnabledFor(logging.INFO):
            logger.info('permutation:%s', self.permutation.to_string())
            logger.info('automorphism:%s', self.automorphism.to_string())

        # apply labeling to a vertices
        self.permutation.apply_vertices(self.vertices)
        return True

    def labeling0(self, partition, check_list, check_start, split, permutation, automorphism, zero_index):
        """Search for labeling by partitioning vertices by potential group of symmetry.

        Args:
            partition (list): a partition of vertices as a list of lists of vertices
            check_list (list): a list of flags for checking shattering of partitions
            check_start (int): start index for checking shattering
            split (list): a list of vertices splitting a partition
            permutation (list): a list of found labeling
            automorphism (AutoMorphism): a list of found automorphisms
            zero_index (int): index of vertices for zero check

        Returns:
            bool: True if success in zero_check
        """
        # make given partition equitable
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('partition:%s', self.print_partition(partition, split))
        refine, check_list = self.equitable_refinement(partition, check_list, check_start)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('refine:%s', self.print_partition(refine, split))

        if self.zero_check and len(refine) > 0:
            if all([v.index != zero_index for v in refine[0]]):
                return False
            self.zero_check = False

        # split not-singleton partition
        for i in range(len(refine)):
            if len(refine[i]) > 1:
                # split partition refine[i] by each vertex
                split_vertices = refine[i]
                split_index = [v.index for v in split_vertices]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('split_vertices:%s', split_index)
                for target in split_vertices:
                    new_partition = refine[0:i]
                    new_partition.append([target])
                    new_partition.append([v for v in split_vertices if v != target])
                    new_partition.extend(refine[i+1:])
                    new_check_list = check_list[0:i]
                    new_check_list.append(True)
                    new_check_list.append(True)
                    new_check_list.extend(check_list[i+1:])
                    split.append(target)
                    self.labeling0(new_partition, new_check_list, i, split, permutation, automorphism, zero_index)
                    if automorphism.fix(split_index):
                        # all the permutation is already generated
                        # prune remaining children
                        break
                return True
        # discrete partition -- reach leaf node
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('discrete:%s', self.print_partition(refine, split))
        # make permutation from discrete partition
        perm = Permutation.from_partition(refine)
        if len(permutation) > 0:
            # make an automorphism from two permutations
            autom = Permutation(len(refine))
            permutation[0].apply_perm(autom)
            perm.r_apply_perm(autom)
            automorphism.add_generator(autom)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('perm1:%s', permutation[0].to_string())
                logger.debug('perm2:%s', perm.to_string())
                logger.debug('auto:%s', autom.to_string_short())
        permutation.append(perm)
        return True

    def equitable_refinement(self, partition, check_list, start):
        """Refine a partition by splitting them until it is shattered.

        Args:
            partition (list): a partition of vertices as a list of lists of vertices
            check_list (list): a list of flags for checking shattering of partitions
            start (int): start index for checking partitions

        Returns:
             list, list: a refined partition, a refined flags
        """
        # make an ordered partition equitable
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('equitable_refine:%s', self.print_partition(partition, []))
        shattering = []
        istart = start
        jstart = 0
        while shattering is not None:
            shattering = None
            for i in range(istart, len(partition)):
                if check_list[i]:
                    for j in range(jstart, len(partition)):
                        shattering = self.shattering(partition[j], partition[i])
                        if shattering is not None:
                            # replace partition[j] with new sub-partition shattering
                            partition[j:j+1] = shattering
                            check_list[j:j+1] = [True]*len(shattering)
                            break
                    if shattering is not None:
                        if j <= i:
                            istart = j
                            jstart = 0
                        else:
                            istart = i
                            jstart = j + len(shattering)
                        break
                    else:
                        check_list[i] = False
                        jstart = 0
        return partition, check_list

    def shattering(self, vi, vj):
        """Check if a list of vertices shatters (vj) another list of vertices (vi).

        Args:
            vi (list): a list of vertices checked shattered
            vj (list): a list of vertices checked shattering

        Returns:
            list: a new partitions shattered by vj
        """
        # check if vj shatter vi (divide vi based on edges to vj)
        # if vi is shattered, create new local partition
        if len(vi) == 1:
            return None
        shattering = defaultdict(list)
        for v in vi:
            shattering[self.degree(v, vj)].append(v)
        if len(shattering) == 1:
            # not shatter
            return None
        else:
            # shatter
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('shattering:%s %s', [x.index for x in vi], [x.index for x in vj])
            return [shattering[key] for key in sorted(shattering.keys())]

    @staticmethod
    def degree(v, vlist):
        """Get the number of degree from a list of vertices to a vertex.

        Args:
            v (AtomVertex): a vertex
            vlist (list): a list of vertices

        Returns:
            int: degrees of a vertex
        """
        # count the number of edge from a vertex v to a set of vertices vlist
        count = Counter()
        for e in v.edges:
            if e.end in vlist:
                count[e.bond_type] += 1
        return tuple(sorted(count.items()))

    def set_label(self):
        """Overwrite a vertex index by its label
        """
        self.permutation.apply_vertices(self.vertices)

    def orbit(self, v):
        """Get an orbit of a vertex in automorphism.

        Returns:
            list: an orbit
        """
        return self.automorphism.orbit(v)

    def is_min_orbit(self, v):
        """Get a minimum index in an orbit of a vertex.

        Returns:
            int: a minimum index
        """
        return self.automorphism.is_min_orbit(v)

    def zero_equivalent(self, index):
        """Check if a vertex index is equivalent to zero labeled vertex.

        Returns:
            bool: True is equivalent. False otherwise.
        """
        if self.automorphism is None:
            if not self.labeling(index):
                return False
        return self.automorphism.equivalent(index, self.permutation.r_apply(0))

    @staticmethod
    def print_partition(partition, split):
        """Get string representation of a partition.

        Args:
            partition (list): a partition of vertices as a list of lists of vertices
            split (list): a list of vertices splitting a partition

        Returns:
            str: string representation
        """
        # print an ordered partition
        rstr = '{'
        for vs in partition:
            rstr += '|'
            for v in vs:
                if v == vs[-1]:
                    rstr += '%d' % v.index
                else:
                    rstr += '%d,' % v.index
        rstr += '|,('
        for v in split:
            if v == split[-1]:
                rstr += '%d' % v.index
            else:
                rstr += '%d,' % v.index
        rstr += ')}'
        return rstr

    def print_vertices(self):
        """Get string representation of graph vertices.

        Returns:
            str: string representation
        """
        # print graph with original vertex index
        rstr = 'ChemGraph:{'
        for v in self.vertices:
            rstr += v.to_string()
            if v != self.vertices[-1]:
                rstr += ','
        rstr += '}'
        return rstr

    def print_vertices_label(self):
        """Get string representation of graph vertices by label

        Returns:
            str: string representation
        """
        # print graph with labelled index
        vertices = sorted(self.vertices, key=lambda x: x.label)
        rstr = 'ChemGraph(cl):{'
        for v in vertices:
            rstr += v.to_string_label()
            if v != vertices[-1]:
                rstr += ','
        rstr += '}'
        return rstr


class Permutation:
    """Permutation of indices starting from 0.

    Attributes:
        permutation (array): permutation of indices
        reverse (array): reverse permutation of indices
    """

    def __init__(self, length):
        """Constructor of permutation of given length.

        Args:
            length (int): length of permutation
        """
        self.permutation = array('l', range(length))
        self.reverse = array('l', range(length))

    @classmethod
    def from_partition(cls, partition):
        """Generate a permutation from a partition. A discrete partition is assumed.

        Args:
            partition (list): a list of lists of vertices

        Returns:
            Permutation: a permutation
        """
        perm = cls(len(partition))
        for index, v in enumerate(partition):
            perm.permutation[v[0].index] = index
            perm.reverse[index] = v[0].index
        return perm

    @classmethod
    def from_index_list(cls, index_list):
        """Generate a permutation from a list of indices

        Args:
            index_list (list): a list of indices representing a permutation
        """
        perm = cls(len(index_list))
        for index in range(len(index_list)):
            perm.permutation[index] = index_list[index]
            perm.reverse[index_list[index]] = index
        return perm

    def get_index_list(self):
        """Get permutation by a list of index

        Returns:
            list: permutation as a list of index
        """
        return list(self.permutation)

    def get_r_index_list(self):
        """Get reverse permutation by a list of index

        Returns:
            list: reverse permutation as a list of index
        """
        return list(self.reverse)

    def apply(self, val):
        """Get a permutation result.

        Args:
            val (int): an index

        Returns:
            int: a permutation of an index
        """
        return self.permutation[val]

    def r_apply(self, val):
        """Get a reverse permutation result.

        Args:
            val (int): an index

        Returns:
            int: a reverse permutation of an index
        """
        return self.reverse[val]

    def apply_vertices(self, vertices):
        """Set permutations of vertex indices to vertex label

        Args:
            vertices (list): a list of vertices
        """
        for v in vertices:
            v.label = self.apply(v.index)

    def r_apply_vertices(self, vertices):
        """Set reverse permutations of vertex indices to vertex label.

        Args:
            vertices (list): a list of vertices
        """
        for v in vertices:
            v.label = self.r_apply(v.index)

    def apply_list(self, vals):
        """Get a permutation results for a list of indices.

        Args:
            vals (list): a list of indices

        Returns:
            list: a list of permutations
        """
        return [self.permutation[x] for x in vals]

    def r_apply_list(self, vals):
        """Get a reverse permutation results for a list of indices.

        Args:
            vals (list): a list of indices

        Returns:
            list: a list of reverse permutations
        """
        return [self.reverse[x] for x in vals]

    def apply_perm(self, perm):
        """Set a permutation to another permutation.

        Args:
            perm (Permutation): a permutation
        """
        for index in range(len(perm.permutation)):
            val = self.apply(perm.permutation[index])
            perm.permutation[index] = val
            perm.reverse[val] = index

    def r_apply_perm(self, perm):
        """Set a reverse permutation to another permutation.

        Args:
            perm (Permutation): a permutation
        """
        for index in range(len(perm.permutation)):
            val = self.r_apply(perm.permutation[index])
            perm.permutation[index] = val
            perm.reverse[val] = index

    def apply_autom(self, autom):
        """Set a permutation to an automorphism.

        Args:
            autom (AutoMorphism): an automorphism
        """
        for perm in autom.generator:
            temp = {}
            for index in range(len(perm.permutation)):
                temp[self.apply(index)] = self.apply(perm.permutation[index])
            for index, val in temp.items():
                perm.permutation[index] = val
                perm.reverse[val] = index

    def to_string(self):
        """Get string representation of a permutation

        Returns:
            str: a string representation
        """
        rstr = '('
        for index in range(len(self.permutation)):
            rstr += '%d' % self.permutation[index]
            if index+1 < len(self.permutation):
                rstr += ','
        rstr += ')'
        return rstr

    def to_string_short(self):
        """Get compact string representation of a permutation.

        Returns:
            str: a string representation
        """
        mark = array('l', [0]*len(self.permutation))
        rstr = ''
        for index in range(len(self.permutation)):
            if mark[index] > 0:
                # skip
                continue
            else:
                if self.permutation[index] == index:
                    mark[index] = 1
                    continue
                else:
                    sub_index = index
                    rstr += '('
                    while True:
                        rstr += '%d' % sub_index
                        mark[sub_index] = 1
                        sub_index = self.permutation[sub_index]
                        if sub_index == index:
                            rstr += ')'
                            break
                        else:
                            rstr += ','
        return rstr


class AutoMorphism:
    """Automorphism of graph labeling by a set of permutations as a generator of permutation group.

    Attributes:
        generator (list): a list of permutations
    """

    def __init__(self):
        """Constructor.
        """
        self.generator = []

    def add_generator(self, perm):
        """Add a generator of permutation group

        Args:
            perm (Permutation): a permutation
        """
        self.generator.append(perm)

    def get_generator(self):
        """Get a list of permutations

        Returns:
            list: a list of permutations
        """
        return self.generator

    def get_generator_by_index_list(self):
        """Get a list of index lists

        Returns:
            list: a list of index lists
        """
        return [perm.get_index_list() for perm in self.generator]

    def orbit_list(self, lst):
        """Get orbits of a list of indices.

        Args:
            lst (list): a list of indices

        Returns:
            a list of orbits of indices
        """
        return [perm.apply_list(lst) for perm in self.generator]

    def orbit(self, index):
        """Get an orbit of an index.

        Args:
            index (int): an index

        Returns:
            list: an orbit
        """
        orbit = set()
        orbit.add(index)
        old_orbit = [index]
        while len(old_orbit) > 0:
            new_orbit = []
            for v in old_orbit:
                for g in self.generator:
                    o = g.apply(v)
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
                    o = g.r_apply(v)
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
            old_orbit = new_orbit
        return orbit

    def is_min_orbit(self, index):
        """Check if an index is a minimum index in its orbit.

        Args:
            index (int): an index

        Returns:
            bool: True is minimum. False otherwise.
        """
        orbit = set()
        orbit.add(index)
        old_orbit = [index]
        while len(old_orbit) > 0:
            new_orbit = []
            for v in old_orbit:
                for g in self.generator:
                    o = g.apply(v)
                    if o < index:
                        return False
                    ro = g.r_apply(v)
                    if ro < index:
                        return False
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
                    if o != ro and ro not in orbit:
                        orbit.add(ro)
                        new_orbit.append(ro)
            old_orbit = new_orbit
        return True

    def min_orbit(self, index):
        """Get a minimum index in an orbit of an index.

        Args:
            index (int): an index

        Returns:
            int: minimum index in an orbit
        """
        orbit = set()
        orbit.add(index)
        old_orbit = [index]
        min_index = index
        while len(old_orbit) > 0:
            new_orbit = []
            for v in old_orbit:
                for g in self.generator:
                    o = g.apply(v)
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
                        if o < min_index:
                            min_index = o
                    o = g.r_apply(v)
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
                        if o < min_index:
                            min_index = o
            old_orbit = new_orbit
        return min_index

    def max_orbit(self, index):
        """Get a maximum index in an orbit of an index.

        Args:
            index (int): an index

        Returns:
            int: maximum index in an orbit
        """
        orbit = set()
        orbit.add(index)
        old_orbit = [index]
        max_index = index
        while len(old_orbit) > 0:
            new_orbit = []
            for v in old_orbit:
                for g in self.generator:
                    o = g.apply(v)
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
                        if o > max_index:
                            max_index = o
                    o = g.r_apply(v)
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
                        if o > max_index:
                            max_index = o
            old_orbit = new_orbit
        return max_index

    def fix(self, vlist):
        """Check if a set of vertices are fixed by an automorphism.

        Args:
            vlist (list): a list of vertices

        Returns:
            bool: True is fixed. False otherwise.
        """
        orbit = self.orbit(vlist[0])
        if len(orbit) >= len(vlist):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('fix:%s %s', orbit, vlist)
            return True
        else:
            return False

    def equivalent(self, index, zero_index):
        """Check if an index is equivalent to another index under an automorphism.

        Args:
            index (int): an index
            zero_index (int): another index

        Returns:
            bool: True if equivalent. False otherwise.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('equivalent:%d %d', index, zero_index)
        if index == zero_index:
            return True
        orbit = set()
        orbit.add(index)
        old_orbit = [index]
        while len(old_orbit) > 0:
            new_orbit = []
            for v in old_orbit:
                for g in self.generator:
                    o = g.apply(v)
                    if o == zero_index:
                        return True
                    ro = g.r_apply(v)
                    if ro == zero_index:
                        return True
                    if o not in orbit:
                        orbit.add(o)
                        new_orbit.append(o)
                    if o != ro and ro not in orbit:
                        orbit.add(ro)
                        new_orbit.append(ro)
            old_orbit = new_orbit
        return False

    def to_string(self):
        """Get string representation of an automorphism
        Returns:
            str: a string representation
        """
        rstr = '['
        for g in self.generator:
            rstr += g.to_string_short()
            if g != self.generator[-1]:
                rstr += ','
        rstr += ']'
        return rstr
