# -*- coding:utf-8 -*-
"""
ChemGraphGen.py

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
from .ChemGraphFragment import *
from .ChemGraphLabeling import *
from .ChemGraphResource import *
from .ChemGraphGenPath import *
from .ChemGraphUtil import NumRange

import numpy as np
import scipy as sp

from collections import Counter, defaultdict
import copy
import math
import sys

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AtomGraphNode(object):
    """Search tree node for generating a graph by adding a vertex one by one in a depth first search manner.
    Since the search algorithm is based on the McKey's canonical construction path algorithm, child nodes
    can be generated independently.

    Attributes:
        atom (str): name of a vertex to add at this search node
        gen_path (str): generation path as a unique id
        pos (AtomVertex): a vertex extending the graph
        vertex (AtomVertex): a vertex added at this search node
        parent (ChemGraphNode): parent node of a search tree
        graph (AtomGraph): current graph (shared by search nodes by default)
        automorphism (ChemGraphLabeling): automorphism of a graph
        atom_count (Counter): counter of atoms in a graph
        atom_degree_count (dict): counter of atom degree in a graph
        ring_count (Counter): counter of rings in a graph
        ring_group_count (Counter): counter of ring groups in a graph
        fragment_count (Counter): counter of fragment occurrence
        fragment_list(list): a list of fragments to count
        num_ring_count (Counter): counter of rings
        num_aromatic_count (Counter): counter aromatic rings
        total_atom_count (int): counter of total atoms
        total_ring_count (int): counter of total rings
        total_aromatic_count (int): counter of total aromatic rings
        online_features (dict): a dictionary of online update features
        feature_values (dict): values of other features than build-in features
        local_fragment_count (Counter): counter of fragments increased in this node
        local_fragment_path (list): a list of matched vertices of a fragment increased in this node
        local_cancel_fragment_path (list): a list of  matched vertices of a fragment cancelled in this node
    """

    def __init__(self, parent, graph, resource, automorphism):
        """Constructor of a search tree node.

        Args:
            parent (ChemGraphNode): parent node of a search tree
            graph (AtomGraph): current graph (shared by search nodes by default)
            resource (GraphGenResourceManager.Resource): resource to add
            automorphism (ChemGraphLabeling): automorphism of a graph
        """
        self.parent = parent
        self.gen_path = ''
        self.expand_index = None
        self.operation = None
        self.resource = resource
        self.atom = ''
        self.vertex = None
        self.pos = (None, None, None)
        self.graph = graph
        self.automorphism = automorphism
        self.depth = 0

        if parent is not None:
            self.atom = resource.get_symbol()
            self.vertex = resource.get_vertex()
            self.depth = parent.depth + 1
            self.atom_count = Counter(parent.atom_count)
            self.atom_degree_count = dict()
            for atom, degree_count in parent.atom_degree_count.items():
                self.atom_degree_count[atom] = np.array(degree_count)
            self.ring_count = Counter(parent.ring_count)
            self.ring_group_count = Counter(parent.ring_group_count)
            self.fragment_count = Counter(parent.fragment_count)
            self.num_ring_count = Counter(parent.num_ring_count)
            self.num_aromatic_count = Counter(parent.num_aromatic_count)
            self.total_atom_count = parent.total_atom_count
            self.total_ring_count = parent.total_ring_count
            self.total_aromatic_count = parent.total_aromatic_count
            self.feature_values = copy.deepcopy(parent.feature_values)
            self.feature_values['amd_tool'] = dict()
            self.online_features = parent.online_features
            self.fragment_list = parent.fragment_list
            # update generation path as id of node
            self.operation = self.parent.get_operation(resource)
            self.gen_path = self.parent.gen_path + self.operation.to_string()
        else:
            self.atom_count = Counter()
            self.atom_degree_count = dict()
            self.ring_count = Counter()
            self.ring_group_count = Counter()
            self.fragment_count = Counter()
            self.num_ring_count = Counter()
            self.num_aromatic_count = Counter()
            self.total_atom_count = 0
            self.total_ring_count = 0
            self.total_aromatic_count = 0
            self.feature_values = defaultdict(dict)
            self.online_features = defaultdict(list)
            self.fragment_list = []
        self.local_fragment_count = Counter()
        self.local_fragment_path = []
        self.local_cancel_fragment_path = []

    def position_to_extend(self):
        """Get available positions of vertices to extend a graph at this node.

        Returns:
            list: a list of extendable vertices
        """
        # find available position to extend the graph
        pos = []
        for v in self.graph.vertices:
            # check valence
            # select only atom with free valence
            if v.num_all_free_hand() <= 0:
                continue
            # check orbit
            # select only one extending position in isomorphic graphs
            if not self.automorphism.is_min_orbit(v.index):
                continue
            pos.append(v)
        return pos

    def get_generation_path(self):
        """Get a generation path with expand position

        Returns:
            str: generation path and expand position
        """
        return self.gen_path

    def get_operation(self, resource):
        """Get generation path from newly added resource

        Args:
            resource (GraphGenResourceManager.Resource): new resource
        """
        (node_pos, bond_type, node_index) = self.pos
        operation = GenerationPath.Operation(self.expand_index,
                                             node_pos.index if node_pos is not None else None,
                                             bond_type, resource, node_index)
        return operation

    def get_op_sequence(self):
        """Get an operation sequence of generation path

        Returns:
            list: a list of operations
        """
        op_sequence = []
        current = self
        while current.parent is not None:
            op_sequence.append(copy.copy(current.operation))
            current = current.parent
        op_sequence.reverse()
        return op_sequence

    def set_online_features(self, online_features):
        """Set a dictionary of online update features

        Args:
            online_features (dict): a dictionary of online update features
        """
        self.online_features = online_features

    def set_fragment_list(self, fragment_list):
        """Set a list of fragments to count

        Args:
            fragment_list (list): a list of fragment to count
        """
        self.fragment_list = fragment_list

    def update_atom_degree_count(self, vertex, new_vertex, bond_type, graph=None):
        """Update degree count of each atom for checking the termination by fp fragment constraint

        Args:
            vertex (AtomVertex): vertex at an extending position
            new_vertex (AtomVertex): new vertex connected at the node
            bond_type (BondType): bond type of connection
            graph (AtomGraph, optional): graph of new vertex. Defaults to None.
        """
        # update atom degree count
        if bond_type is not None:
            bond_order = ChemGraph.get_bond_order(bond_type)
        else:
            bond_order = 0
        # update degree count by vertex at an extending position
        if vertex is not None:
            if isinstance(vertex, ConnectionVertex):
                # new bond is not yet set to real vertex
                real_vertex = vertex.connect
                bond_degree = real_vertex.bond_degree()
                max_degree = len(self.atom_degree_count[real_vertex.atom]) - 1
                self.atom_degree_count[real_vertex.atom][min(int(bond_degree + bond_order), max_degree)] += 1
                self.atom_degree_count[real_vertex.atom][bond_degree] -= 1
            else:
                # new bond is already set to real vertex
                bond_degree = vertex.bond_degree()
                self.atom_degree_count[vertex.atom][bond_degree] += 1
                self.atom_degree_count[vertex.atom][int(bond_degree - bond_order)] -= 1
        # update degree count by vertices in graph of new vertex
        if graph is not None:
            for atom, ring_atom_degree in graph.atom_degree_count.items():
                atom_degree_count = self.atom_degree_count[atom]
                for index in range(1, len(ring_atom_degree)):
                    atom_degree_count[0] -= ring_atom_degree[index]
                    atom_degree_count[index] += ring_atom_degree[index]
        # update degree count by newly added vertex
        if vertex is not None and new_vertex is not None:
            if isinstance(new_vertex, SubStructureAsVertex):
                # new bond is not yet set to real vertex
                real_vertex = new_vertex.connection_vertex.connect
                bond_degree = real_vertex.bond_degree()
                max_degree = len(self.atom_degree_count[real_vertex.atom]) - 1
                self.atom_degree_count[real_vertex.atom][min(int(bond_degree + bond_order), max_degree)] += 1
                self.atom_degree_count[real_vertex.atom][bond_degree] -= 1
            else:
                # new bond is already set to real vertex
                bond_degree = new_vertex.bond_degree()
                self.atom_degree_count[new_vertex.atom][bond_degree] += 1
                self.atom_degree_count[new_vertex.atom][int(bond_degree - bond_order)] -= 1

    def extend_by_atom(self, pos, atom_resource, bond_type, active_node, canonicity_check):
        """Extend a graph by adding an atom, and check the feasibility of canonical construction path.

        Args:
            pos (AtomVertex): a vertex to add a new atom vertex
            atom_resource (AtomResManager.Resource): atom resource to add
            bond_type (BondType): bond type of an edge connecting a new vertex
            active_node (dict): a dictionary of active nodes in beam size
            canonicity_check (bool): a flag of canonical path check

        Returns:
            AtomGraphNode: a new node if feasible. Otherwise, None.
        """
        # extend the node by adding an atom if feasible
        atom_vertex = atom_resource.new_instance()
        new_v = self.graph.add_vertex(atom_vertex.atom)
        # update valence of new vertex
        new_v.valence = atom_vertex.valence
        if pos is not None:
            self.graph.add_edge(new_v, pos, bond_type)
            self.pos = (pos, bond_type, None)
            local_cancel_fragment_path = [x for x in pos.get_exact_match_fragment()]
        else:
            self.pos = (None, bond_type, None)
            local_cancel_fragment_path = []

        if logger.isEnabledFor(logging.INFO):
            logger.info('connection vertex: {0}'.format(self.graph.to_string()))

        if active_node is None:
            # check canonical path
            if canonicity_check:
                new_labeling = ChemGraphLabeling(self.graph.vertices, zero_check=True)
                if new_labeling.zero_equivalent(new_v.index):
                    graph_automorphism = new_labeling.automorphism
                else:
                    self.graph.pop_vertex()
                    atom_resource.save_instance()
                    return None
            else:
                new_labeling = ChemGraphLabeling(self.graph.vertices)
                graph_automorphism = new_labeling.automorphism
                if not new_labeling.zero_equivalent(new_v.index):
                    operation = self.get_operation(atom_resource)
                    logger.warning('not a canonical path: path={0} op={1} {2}'.
                                   format(self.gen_path, operation.to_string(),
                                          self.graph.to_string()))
        else:
            # check active node
            operation = self.get_operation(atom_resource)
            new_generation_path = self.gen_path + operation.to_string()
            if new_generation_path in active_node:
                graph_automorphism = active_node[new_generation_path]
            else:
                self.graph.pop_vertex()
                atom_resource.save_instance()
                return None

        # create new search node
        new_node = AtomGraphNode(self, self.graph, atom_resource, graph_automorphism)
        # update resource count
        new_node.atom_count += atom_resource.atom_count
        new_node.total_atom_count += 1
        # update atom degree count
        new_node.update_atom_degree_count(pos, new_v, bond_type)
        # update fragment count
        new_node.update_feature_values(new_v)
        # update canceled fragment count
        new_node.local_cancel_fragment_path = local_cancel_fragment_path
        # update accumulated fragment count
        new_node.fragment_count += new_node.local_fragment_count
        for fpath in local_cancel_fragment_path:
            (f, path) = fpath
            new_node.fragment_count[f] -= 1
            # cancel fragment path from vertices in the path
            for v in path:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('extend remove fpath:v={0} fragment={1}, path={2}'.format(v.index,
                                                                                           f.to_string(),
                                                                                           [v0.index for v0 in path]))
                v.exact_match_fragment.remove(fpath)
            if logger.isEnabledFor(logging.DEBUG):
                for v in new_node.graph.vertices:
                    for fpath2 in v.exact_match_fragment:
                        (f, path) = fpath2
                        logger.debugo('saved fpath:v={0} fragment={1} path={2}'.
                                      format(v.index, f.to_string(), [v0.index for v0 in path]))
            if logger.isEnabledFor(logging.INFO):
                for f, count in new_node.fragment_count.items():
                    logger.info('fragment count:{0} {1}'.format(count, f.to_string()))
        return new_node

    def extend_by_ring(self, pos, ring_resource, index, bond_type, active_node, canonicity_check):
        """Extend a graph by adding a ring, and check the feasibility of canonical construction path.

        Args:
            pos (AtomVertex): a vertex to add a new atom vertex
            ring_resource (RingResManager.Resource): ring resource to add
            index (int): a vertex index of a ring to connect to a graph
            bond_type (BondType): bond type of an edge connecting a new vertex
            active_node (dict): a dictionary of active nodes in a beam search
            canonicity_check (bool): a flag of canonical path check

        Returns:
            AtomGraphNode: a new node if feasible. Otherwise, None.
        """
        # extend the node by adding a ring if feasible
        ring_graph = ring_resource.new_instance()
        extension = ring_resource.get_extension()
        automorphism = ring_resource.get_labeling().automorphism if pos is None else extension[index][1]
        ring_graph.set_automorphism(automorphism)
        new_v = self.graph.add_ring_graph_vertex_by_graph(ring_graph, index)
        new_real_v = new_v.connection_vertex.connect
        if pos is not None:
            self.graph.add_edge(new_v, pos)
            self.graph.add_edge(new_v.connection_vertex, pos, bond_type)
            self.pos = (pos, bond_type, index)
            local_cancel_fragment_path = [x for x in pos.get_exact_match_fragment()]
            local_cancel_fragment_path.extend([x for x in new_real_v.get_exact_match_fragment()])
        else:
            self.pos = (None, bond_type, index)
            local_cancel_fragment_path = []

        if logger.isEnabledFor(logging.INFO):
            logger.info('connection vertex: {0}'.format(self.graph.to_string()))

        if active_node is None:
            # check canonical path
            if canonicity_check:
                new_labeling = ChemGraphLabeling(self.graph.vertices, zero_check=True)
                if new_labeling.zero_equivalent(new_v.index):
                    graph_automorphism = new_labeling.automorphism
                else:
                    self.graph.pop_ring_graph_vertex()
                    ring_graph.set_automorphism(None)
                    ring_resource.save_instance()
                    return None
            else:
                new_labeling = ChemGraphLabeling(self.graph.vertices)
                graph_automorphism = new_labeling.automorphism
                if not new_labeling.zero_equivalent(new_v.index):
                    operation = self.get_operation(ring_resource)
                    logger.warning('not a canonical path: path={0} op={1} {2}'.
                                   format(self.gen_path, operation.to_string(),
                                          self.graph.to_string()))
        else:
            # check active node
            operation = self.get_operation(ring_resource)
            new_generation_path = self.gen_path + operation.to_string()
            if new_generation_path in active_node:
                graph_automorphism = active_node[new_generation_path]
            else:
                self.graph.pop_ring_graph_vertex()
                ring_graph.set_automorphism(None)
                ring_resource.save_instance()
                return None

        # create new search node
        new_node = AtomGraphNode(self, self.graph, ring_resource, graph_automorphism)
        # update resource count
        new_node.atom_count += ring_resource.atom_count
        new_node.total_atom_count += sum(ring_resource.atom_count.values())
        # pp update atom degree count
        new_node.update_atom_degree_count(pos, new_v, bond_type, ring_graph)
        # update ring count
        new_node.ring_count += Counter([ring_resource.get_symbol()])
        new_node.ring_group_count += Counter([ring_resource.get_base_symbol()])
        new_node.num_ring_count += new_v.ring_count
        new_node.num_aromatic_count += new_v.aromatic_count
        new_node.total_ring_count += new_v.total_ring_count
        new_node.total_aromatic_count += new_v.total_aromatic_count
        # update fragment count
        new_node.update_feature_values(new_v)
        # update canceled fragment count
        new_node.local_cancel_fragment_path = local_cancel_fragment_path
        # update accumulated fragment count
        new_node.fragment_count += new_node.local_fragment_count
        for fpath in local_cancel_fragment_path:
            (f, path) = fpath
            new_node.fragment_count[f] -= 1
            # cancel fragment path from vertices in the path
            for v in path:
                if logger.isEnabledFor(logging.INFO):
                    logger.debug('extend remove fpath:v={0} fragment={1}, path={2}'.format(v.index, f.to_string(),
                                                                                           [v0.index for v0 in path]))
                v.exact_match_fragment.remove(fpath)
        if logger.isEnabledFor(logging.DEBUG):
            for v in new_node.graph.vertices:
                for fpath in v.exact_match_fragment:
                    (f, path) = fpath
                    logger.debug('saved fpath:v={0} fragment={1} path={2}'.format(v.index, f.to_string(),
                                                                                  [v0.index for v0 in path]))
        if logger.isEnabledFor(logging.INFO):
            for f, count in new_node.fragment_count.items():
                logger.info('fragment count:{0} {1}'.format(count, f.to_string()))
        return new_node

    def update_feature_values(self, new_v):
        """Update feature values by a new vertex.

        Args:
            new_v (AtomVertex): a new vertex
        """
        # count the increase of the fragments by adding new vertex
        if len(self.fragment_list) == 0 and len(self.online_features) == 0:
            return

        # get extended vertex
        pos_vertex = None if self.parent is None else self.parent.pos[0]
        if isinstance(pos_vertex, ConnectionVertex):
            pos_vertex = pos_vertex.get_connect_vertex()
        # get extending vertex
        if isinstance(new_v, SubStructureAsVertex):
            new_vertex = new_v.connection_vertex.get_connect_vertex()
        else:
            new_vertex = new_v

        # get automorphism of graph to expand without labelling
        expand_automorphism = self.graph.get_expand_automorphism(self.automorphism)
        # expand all the shrunk vertices
        self.graph.expand_graph()
        # make min_orbit map of vertex index of a graph
        if logger.isEnabledFor(logging.INFO):
            logger.info('count_fragment:{0}'.format(self.graph.to_string()))
        orbit_map = [expand_automorphism.min_orbit(v.index) for v in self.graph.vertices]

        # verify orbit_map in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            labeling = ChemGraphLabeling(self.graph.vertices)
            debug_orbit_map = [labeling.automorphism.min_orbit(v.index) for v in self.graph.vertices]
            if orbit_map != debug_orbit_map:
                logger.debug('expand_graph:{0}'.format(self.graph.to_string()))
                logger.debug('old autom:{0}'.format(labeling.automorphism.to_string()))
                logger.debug('new autom:{0}'.format(expand_automorphism.to_string()))
                logger.debug('old orbit_map:{0}'.format(debug_orbit_map))
                logger.debug('new orbit_map:{0}'.format(orbit_map))

        # update fragment count by counting fragments at the connection
        if pos_vertex is not None:
            for fragment in self.fragment_list:
                if fragment.enough_atom(self.atom_count):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('check_fragment:%s', fragment.to_string())
                    count, path = fragment.count_fragment_edge(pos_vertex, new_vertex, orbit_map)
                    self.local_fragment_count[fragment] = count
                    if fragment.should_mark_vertex():
                        fragment_path = fragment.mark_vertex(path)
                        self.local_fragment_path.extend(fragment_path)
        # add fragments in the resource object
        self.local_fragment_count += self.resource.fragment_count

        # update feature values
        updated = set()
        for features, feature_list in self.online_features.items():
            if features.is_online_update():
                extractor = features.get_extractor()
                extractor.update_feature_value(self.graph, new_vertex, feature_list, self, updated)

        # shrink ring graph vertices
        self.graph.shrink_graph()

    def restore(self):
        """Restore the state of graph before returning to parent node.

        Returns:
            AtomVertex: a removed vertex
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('restore fragment path:v=%d atom=%s', self.vertex.index, self.atom)
        # restore the state of graph, which is shared by other search nodes
        for fpath in self.local_cancel_fragment_path:
            (f, path) = fpath
            for v in path:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('restore add fpath:v=%d fragment=%s, path=%s',
                                 v.index, f.to_string(), [v0.index for v0 in path])
                v.exact_match_fragment.add(fpath)
        for fpath in self.local_fragment_path:
            (f, path) = fpath
            for v in path:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('restore remove fpath:v=%d fragment=%s, path=%s',
                                 v.index, f.to_string(), [v0.index for v0 in path])
                v.exact_match_fragment.remove(fpath)

        # restore atom/ring/graph vertex
        if isinstance(self.vertex, AtomVertex):
            self.graph.pop_vertex()
        elif isinstance(self.vertex, RingAsVertex):
            self.graph.pop_ring_graph_vertex()

        # save instance to a resource object
        self.resource.save_instance()

    def expand_graph_vertex(self, graph_vertex):
        """Expand a given graph vertex.

        Args:
            graph_vertex (ConnectionVertex): a graph vertex
        """
        if isinstance(graph_vertex, SubStructureAsVertex):
            self.expand_index = graph_vertex.index
        self.graph.expand_graph_vertex(graph_vertex)

    def shrink_graph_vertex(self):
        """Shrink expanded graph vertex.
        """
        self.expand_index = None
        self.graph.shrink_graph_vertex()


class ChemGraphConstraint(object):
    """Constraints of the structure generation as ranges of atoms, rings, fragments. Feasibility of
    molecule structure is checked against the constraint in the structure generation steps.

    Attributes:
        id (str): constraint id
        atom_constraint (dict): a dictionary of feasible ranges of atom definitions
        ring_constraint (dict): a dictionary of feasible ranges of ring definitions
        fragment_constraint (dict): a dictionary of feasible ranges of fragment definitions
        prediction_error (float): acceptable range of prediction error
        molecule_evaluator (MoleculeEvaluator): an evaluator of feature vector of a molecule
        fragments_in_feature_vector (set): a set of fragments used in a feature vector
        atom_resource (dict): a dictionary of feasible ranges of atoms
        total_atom_count (int): sum of max count of atoms
        ring_resource (dict): a dictionary of feasible ranges of rings
        ring_group_resource (dict): a dictionary of feasible ranges of ring groups
        fragment_const (dict): a dictionary of feasible ranges of fragment occurrences
        online_features (dict): a dictionary of online update feature sets
        prohibited_connections1 (dict): a dictionary of prohibited atom connections
        prohibited_connections2 (dict): a dictionary of prohibited edge connections
        ring_range (dict): feasible range of the number of rings
        aromatic_range (NumRange): feasible range of the number of aromatic rings
        total_atom_range (NumRange): feasible range of the number of total atoms
        total_ring_range (NumRange): feasible range of the number of total rings
        total_aromatic_range (NumRange): feasible range of the number of total aromatic rings
        enough_atom (bool): flag if there is enough atom resources
        enough_ring (bool): flag if there is enough ring resources
    """

    invalid_node_score = 1000000
    """score of invalid node (large number)"""

    def __init__(self, id, atom_constraint, ring_constraint, fragment_constraint,
                 online_features,
                 ring_range=None, aromatic_range=None,
                 total_atom_range=None, total_ring_range=None, total_aromatic_range=None, 
                 prediction_error=1.0, molecule_evaluator=None):
        """Constructor of molecular graph constraint.

        Args:
            id (str): constraint id
            atom_constraint (dict): a dictionary of feasible ranges of atom definitions
            ring_constraint (dict): a dictionary of feasible ranges of ring definitions
            fragment_constraint (dict): a dictionary of feasible ranges of fragment definitions
            online_features (dict): a dictionary of online update feature sets
            ring_range (dict, optional): feasible range of the number of rings. Defaults to None.
            aromatic_range (dict, optional): feasible range of the number of aromatic rings. 
                Defaults to None.
            total_atom_range (list, optional): feasible range of the number of total atoms. 
                Defaults to None.
            total_ring_range (list, optional): feasible range of the number of total rings.
                Defaults to None.
            total_aromatic_range(list, optional): feasible range of the number of total aromatic rings. 
                Defaults to None.
            prediction_error(float, optional): acceptable range of prediction error. Defaults to 1.0.
            molecule_evaluator(MoleculeEvaluation, optional): molecule feature vector evaluator.
                Defaults to None.
        """
        self.id = id
        self.atom_constraint = atom_constraint
        self.ring_constraint = ring_constraint
        self.fragment_constraint = fragment_constraint
        self.online_features = online_features
        self.prediction_error = prediction_error
        self.molecule_evaluator = molecule_evaluator
        self.fragments_in_feature_vector = set()

        self.infeasible = False
        self.infeasible_reason = ''
        self.atom_resource = {}
        self.total_atom_count = 0
        self.ring_resource = {}
        self.ring_group_resource = {}
        self.fragment_const = {}
        self.prohibited_connections1 = defaultdict(set)
        self.prohibited_connections2 = defaultdict(set)
        self.ring_range = {}
        self.aromatic_range = {}
        self.total_atom_range = None
        self.total_ring_range = None
        self.total_aromatic_range = None
        self.enough_atom = True
        self.enough_ring = True

        if ring_range is not None:
            for rnum, rrange in ring_range.items():
                self.ring_range[rnum] = NumRange(rrange)
        if aromatic_range is not None:
            for rnum, rrange in aromatic_range.items():
                self.aromatic_range[rnum] = NumRange(rrange)
        if total_atom_range is not None:
            self.total_atom_range = NumRange(total_atom_range)
        if total_ring_range is not None:
            self.total_ring_range = NumRange(total_ring_range)
        if total_aromatic_range is not None:
            self.total_aromatic_range = NumRange(total_aromatic_range)

    def get_id(self):
        """Get resource constraint id

        Returns:
            str: id of resource constraint
        """
        return self.id

    def get_fragments_in_feature_vector(self):
        """Get a set of fragments referred in a feature vector

        Returns:
            set: a set of fragments
        """
        if self.molecule_evaluator is None:
            return set()
        else:
            return self.molecule_evaluator.get_fragments_in_feature_vector()

    def set_infeasible(self, reason):
        """Set infeasible status with reason

        Args:
            reason (str): reason of the infeasibility
        """
        self.infeasible = True
        self.infeasible_reason = reason

    def reset_infeasible(self, reason):
        """Reset infeasible status with reason

        Args:
            reason (str): reason of the infeasibility
        """
        self.infeasible = False
        self.infeasible_reason = ''

    def is_infeasible(self):
        """Check if the constraint is infeasible

        Returns:
            bool: True if infeasible
        """
        return self.infeasible

    def get_reason(self):
        """Get a reason string of the infeasibility

        Returns:
            str: reason string
        """
        return self.infeasible_reason

    def get_resource_definitions(self):
        """Get definitions of resources (atom, ring, graph, fragment).

        Returns:
            set, set, set, set: definition of atom/ring/graph/fragment resources
        """
        atom_definitions = set(self.atom_constraint.keys())
        ring_definitions = set(self.ring_constraint.keys())
        fragement_definitions = set(self.fragment_constraint.keys())
        return atom_definitions, ring_definitions, fragement_definitions

    def set_resource_constraint(self, generator):
        """Initialize constraints for the resources

        Args:
            generator (ChemGraphGenerator): structure generator
        """
        self.set_atom_resource(generator, self.atom_constraint)
        self.set_ring_resource(generator, self.ring_constraint)
        self.set_fragment_constraint(generator, self.fragment_constraint)

        # set up prohibited connection
        connection_fragment = []
        for fragment, frange in self.fragment_const.items():
            if frange.min == 0 and frange.max == 0 and fragment.root_vertex is None:
                if fragment.graph.num_vertices() == 2:
                    # set prohibited connection1
                    atoms = sorted(fragment.graph.vertices, key=lambda x: x.atom)
                    if atoms[0].atom == ChemVertex.wild_card_atom:
                        continue
                    atom0 = atoms[0].atom
                    atom1 = atoms[1].atom
                    bond_type = atoms[0].edges[0].bond_type
                    connection_fragment.append(fragment)
                    self.prohibited_connections1[(atom0, bond_type)].add(atom1)
                elif fragment.graph.num_vertices() == 3:
                    # set prohibited connection2
                    atoms = sorted(fragment.graph.vertices, key=lambda x: (x.num_edge(), x.atom))
                    if atoms[0].num_edge() > 1:
                        continue
                    if atoms[0].atom == ChemVertex.wild_card_atom or \
                            atoms[2].atom == ChemVertex.wild_card_atom:
                        continue
                    atom0 = atoms[0].atom
                    atom1 = atoms[1].atom
                    atom2 = atoms[2].atom
                    bond_types = [atoms[0].edges[0].bond_type, atoms[1].edges[0].bond_type]
                    if atom0 == atom1:
                        bond_types = sorted(bond_types)
                    connection_fragment.append(fragment)
                    self.prohibited_connections2[(atom0, bond_types[0])].\
                        add((atom2, bond_types[1], atom1))

        # remove fragments for prohibited connection from fragments to count in the generation
        self.fragments_in_feature_vector = self.get_fragments_in_feature_vector()
        for fragment in connection_fragment:
            if fragment not in self.fragments_in_feature_vector:
                del self.fragment_const[fragment]

        # check resource constraint by fragment constraint
        self.check_by_fragment_constraint(generator)

    def check_terminate(self, generator, node, available_atom, available_ring):
        """Check the termination of search node by considering remaining atoms and unsatisfied constraints

        Args:
            generator (ChemGraphGenerator): the structure generator
            node (AtomGraphNode): search node of the generator
            available_atom (list): a list of available atoms
            available_ring (list): a list of available rings

        Returns:
            ChemGraphConstraint, list, list, list, list: infeasible constraint object, list of atoms, list of rings,
                list of graphs, list of candidate atoms for next search
        """
        if self.infeasible:
            return None, [], [], [], []

        # count remaining atoms
        remaining_atom0 = Counter()
        remaining_atom1 = Counter()
        for atom in available_atom:
            remaining_atom0[atom] = self.atom_resource[atom].max - node.atom_count[atom]
            remaining_atom1[atom] = self.atom_resource[atom].max - node.atom_count[atom]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('remaining atom: {0}'.format(remaining_atom0))

        # update available atom
        available_atom = \
            self.update_available_atom_resource(generator, node, remaining_atom0, available_atom)
        # update available ring
        available_ring = \
            self.update_available_ring_resource(generator, node, remaining_atom0, available_ring)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('remaining atom0: {0}'.format(remaining_atom0))

        terminate, reason = self.check_terminate_ring(generator, node, available_ring, remaining_atom0)
        terminate, reason = (terminate, reason) if terminate else \
            self.check_terminate_fragment(generator, node)
        terminate, reason = (terminate, reason) if terminate else \
            self.check_terminate_counter_range(generator, node, available_ring, remaining_atom1)

        if terminate:
            self.set_infeasible(reason)
            return self, [], [], []

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('remaining atom0: {0} atom1: {1}'.format(remaining_atom0, remaining_atom1))

        # update available atom for generation from atom
        atom_candidate = [atom for atom in available_atom
                          if min(remaining_atom0[atom], remaining_atom1[atom]) >= 1]

        return None, available_atom, available_ring, atom_candidate

    def check_resource_satisfaction(self, node):
        """Check if the constraints are satisfied at a given search node.

        Args:
            node (AtomGraphNode): a search node

        Returns:
            set: a set of ids of satisfied constraint object
        """
        satisfied_constraint = set()
        if self.infeasible:
            return satisfied_constraint
        if not self.check_counter_ranges(node):
            return satisfied_constraint
        if not self.check_atom_ranges(node):
            return satisfied_constraint
        if not self.check_ring_ranges(node):
            return satisfied_constraint
        if not self.check_fragment_ranges(node):
            return satisfied_constraint
        satisfied_constraint.add(self.id)
        return satisfied_constraint

    def check_non_feature_satisfaction(self, node):
        """Check if the constraint not related to feature vector are satisfied

        Args:
            node (AtomGraphNode): a search node

        Returns:
            bool: true if satisfied
        """

        # check fragment satisfaction
        for fragment, num_range in self.fragment_const.items():
            if fragment in self.fragments_in_feature_vector:
                continue
            node_fragment_count = node.fragment_count[fragment]
            if node_fragment_count > num_range.max:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('fragment exceed:count={0} {1}'.
                                format(node_fragment_count, fragment.to_string()))
                return False
            if node_fragment_count < num_range.min:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('fragment short:count={0} {1}'.
                                format(node_fragment_count, fragment.to_string()))
                return False
        return True

    def get_online_features(self):
        """Get a dictionary of online update features

        Returns:
            dict: a dictionary of online update features
        """
        return self.online_features

    def get_fragments_to_count(self):
        """Get fragment which should be counted in the structure generation

        Returns:
            list: a list of fragment objects to count
        """
        return list(self.fragment_const.keys())

    def update_root_atom_candidate(self, atom_candidate):
        """Update atom candidate for root node

        Args:
            atom_candidate (list): a list of candidate atoms

        Returns:
            list: updated list of candidate atoms
        """
        # get must-be-root atom from fp-structure
        must_be_root_atom = ('', 0)
        for fragment, num_range in self.fragment_const.items():
            if num_range.max == 0:
                continue
            if fragment.root_vertex is not None and fragment.root_vertex.num_edge() == 1:
                v = fragment.root_vertex
                if (v.color(), v.num_bond2()) > must_be_root_atom:
                    must_be_root_atom = (v.color(), v.num_bond2())
        new_atom_candidate = [atom for atom in atom_candidate if atom >= must_be_root_atom[0]]
        return new_atom_candidate

    def is_mandatory_atom_resource(self, atom_symbol):
        """Check if using an atom is mandatory in the constraint

        Args:
            atom_symbol (str): symbol of atom resource

        Returns:
            bool: true if mandatory
        """
        return self.atom_resource[atom_symbol].min > 0

    def is_mandatory_ring_resource(self, ring_symbol):
        """Check if using a ring is mandatory in the constraint

        Args:
            ring_symbol (str): symbol of ring resource

        Returns:
            bool: true if mandatory
        """
        return self.ring_resource[ring_symbol].min > 0

    def set_atom_resource(self, generator, atom_constraint):
        """Set up feasible atom resources.

        Args:
            generator (ChemGraphGenerator): structure generator
            atom_constraint (dict): a dictionary of feasible ranges of atom resources
        """
        # check for atom resources
        for atom_def, arange in atom_constraint.items():
            atom_resource = generator.atom_res_mgr.get_resource_from_definition(atom_def)
            atom_range = NumRange(arange)
            if atom_resource is None:
                continue
            atom_vertex = atom_resource.get_vertex()
            atom_symbol = atom_resource.get_symbol()
            if logger.isEnabledFor(logging.INFO):
                logger.info('set atom resource: {0}:{1} valence={2}'.
                            format(atom_symbol, atom_range, atom_vertex.num_valence()))
            self.atom_resource[atom_symbol] = atom_range
        # set total atom count
        self.total_atom_count = sum([nrange.max for nrange in self.atom_resource.values()])

    def set_ring_resource(self, generator, ring_constraint):
        """Set up feasible ring resources.

        Args:
            generator (ChemGraphGenerator): structure generator
            ring_constraint (dict): a dictionary of feasible ranges of ring resource
        """
        # check for ring resources
        for ring_def, rrange in ring_constraint.items():
            ring_resource = generator.ring_res_mgr.get_resource_from_definition(ring_def)
            ring_range = NumRange(rrange)
            if ring_resource is None:
                continue
            ring_vertex = ring_resource.get_vertex()
            ring_symbol = ring_resource.get_symbol()
            # check feasibility of the number of rings
            infeasible = False
            for ring_size, count in ring_vertex.ring_count.items():
                if ring_size in self.ring_range and self.ring_range[ring_size].max < count:
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('infeasible ring resource: {0} ring={1}'.format(ring_symbol, ring_size))
                    infeasible = True
                    break
            ring_range = NumRange([0, 0]) if infeasible else ring_range
            # set resource range
            if ring_resource.is_base_ring():
                if logger.isEnabledFor(logging.INFO):
                    logger.info('set base ring resource: {0}:{1} {2}'.
                                format(ring_symbol, ring_range, ring_vertex.to_string()))
                if ring_symbol in self.ring_group_resource:
                    new_range = self.ring_group_resource[ring_symbol].intersection(ring_range)
                    logger.info('duplicated ring resource: {0} {1}'.format(member_symbol, new_range))
                    ring_range = NumRange([0, 0]) if new_range is None else new_range
                self.ring_group_resource[ring_symbol] = ring_range
                # set member ring resource
                for member in ring_resource.get_members():
                    member_vertex = member.get_vertex()
                    member_symbol = member.get_symbol()
                    # check feasibility of the number of aromatic rings
                    infeasible = False
                    for ring_size, count in member_vertex.aromatic_count.items():
                        if ring_size in self.aromatic_range and self.aromatic_range[ring_size].max < count:
                            if logger.isEnabledFor(logging.INFO):
                                logger.info('infeasible member resource: {0} ring={1}'.format(member_symbol, ring_size))
                            infeasible = True
                            break
                    member_range = NumRange([0, 0]) if infeasible else NumRange([0, rrange[1]])
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('set member ring resource: {0}:{1} {2}'.
                                    format(member_symbol, member_range, member_vertex.to_string()))
                    if member_symbol in self.ring_resource:
                        new_range = self.ring_resource[member_symbol].intersection(member_range)
                        logger.info('duplicated ring resource: {0} {1}'.format(member_symbol, new_range))
                        member_range = NumRange([0, 0]) if new_range is None else new_range
                    self.ring_resource[member_symbol] = member_range
            else:
                # check feasibility of the number of aromatic rings
                infeasible = False
                for ring_size, count in ring_vertex.aromatic_count.items():
                    if ring_size in self.aromatic_range and self.aromatic_range[ring_size].max < count:
                        logger.info('infeasible ring resource: {0} aromatic={1}'.format(ring_symbol, ring_size))
                        infeasible = True
                        break
                ring_range = NumRange([0, 0]) if infeasible else ring_range
                if logger.isEnabledFor(logging.INFO):
                    logger.info('set ring resource: {0}:{1} {2}'.
                                format(ring_symbol, ring_range, ring_vertex.to_string()))
                if ring_symbol in self.ring_resource:
                    new_range = self.ring_resource[ring_symbol].intersection(rrange)
                    logger.info('duplicated ring resource: {0} {1}'.format(member_symbol, new_range))
                    ring_range = NumRange([0, 0]) if new_range is None else new_range
                self.ring_resource[ring_symbol] = ring_range

    def set_fragment_constraint(self, generator, fragment_constraint):
        """Set up feasible fragment constraints.

        Args:
            generator (ChemGraphGenerator): structure generator
            fragment_constraint (dict): a dictionary of fragment definition and feasible range
        """
        # update generate fragment graph
        for fragment_def, frange in fragment_constraint.items():
            fragment = generator.get_fragment_object(fragment_def)
            if fragment is None:
                continue
            # set fragment constraint
            self.fragment_const[fragment] = NumRange(frange)

    def check_by_fragment_constraint(self, generator):
        """Check the feasibility of the ranges of resource (atom, ring, and sub-structures) by referring to
        the ranges of fragment occurrences, and adjust the ranges.

        Args:
            generator (ChemGraphGenerator): structure generator
        """
        # check satisfiability of atom resources
        self.enough_atom = True
        for atom_symbol, arange in self.atom_resource.items():
            atom_resource = generator.atom_res_mgr.get_resource(atom_symbol)
            for fragment, frange in self.fragment_const.items():
                for fatom, facount in fragment.atom_count.items():
                    if atom_symbol == fatom and frange.min > 0 and arange.max < facount:
                        # inconsistency, not enough atom
                        logger.error('not enough atom %s for fragment %s',
                                     atom_symbol, generator.fragment_smiles[fragment])
                        self.enough_atom = False
                if arange.max == 0:
                    break
                # count a fragment in an atom
                if fragment.graph.num_vertices() == 1:
                    if atom_symbol == fragment.graph.vertices[0].atom:
                        atom_resource.fragment_count[fragment] = 1
                        if arange.max > frange.max:
                            # atom can be used up to frange.max
                            logger.info('atom {0} max usage changed from {1} to {2} due to fragment {3}:count={4}, max={5}'.
                                        format(atom_symbol, arange.max, frange.max,
                                               generator.fragment_smiles[fragment], 1, frange.max))
                            arange.min = min(arange.min, frange.max)
                            arange.max = frange.max

        # check satisfiability of ring resource
        for ring_symbol, rrange in self.ring_resource.items():
            if rrange.max == 0:
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('check_by_fragment:ring={0}'.format(ring_symbol))
            ring_resource = generator.ring_res_mgr.get_resource(ring_symbol)
            ring_graph = ring_resource.get_vertex()
            ring_label = ring_resource.get_labeling()
            # check by prohibited connection
            if rrange.max > 0 and not self.check_prohibited_connection_of_graph(ring_graph):
                logger.info('ring {0} max usage changed from {1} to {2} due to prohibited connection'.
                            format(ring_symbol, rrange.max, 0))
                rrange.min = 0
                rrange.max = 0
            # check by atom resource
            if rrange.max > 0:
                for atom, count in ring_resource.atom_count.items():
                    max_use = int(self.atom_resource[atom].max/count)
                    if rrange.max > max_use:
                        # ring can be used up to max_use
                        logger.info('ring {0} max usage changed from {1} to {2} due to atom {3}'.
                                    format(ring_symbol, rrange.max, max_use, atom))
                        rrange.min = min(rrange.min, max_use)
                        rrange.max = max_use
            # check by fragment const
            for fragment, frange in self.fragment_const.items():
                if rrange.max == 0:
                    break
                # count a fragment in a ring
                count, path = fragment.count_fragment_graph_with_path(ring_graph, ring_label)
                if count > 0:
                    ring_resource.fragment_count[fragment] = count
                    if fragment.should_mark_vertex():
                        ring_resource.fragment_path.extend(fragment.mark_vertex(path))
                # update a range of a ring
                if fragment.exact_match:
                    continue
                if fragment.root_vertex is not None and not fragment.saturated_root():
                    continue
                if count > 0 and rrange.max > int(frange.max/count):
                    # check maximum use of the ring
                    max_use = int(frange.max/count)
                    # ring can be used up to max_use
                    logger.info('ring {0} max usage changed from {1} to {2} due to fragment {3}:count={4}, max={5}'.
                                format(ring_symbol, rrange.max, max_use,
                                       generator.fragment_smiles[fragment], count, frange.max))
                    rrange.min = min(rrange.min, max_use)
                    rrange.max = max_use

        # check satisfiability of ring group resource
        for base_symbol, rrange in self.ring_group_resource.items():
            base_resource = generator.ring_res_mgr.get_base_resource(base_symbol)
            total_count = 0
            for member in base_resource.get_members():
                total_count += self.ring_resource[member.get_symbol()].max
            if rrange.min > total_count:
                logger.error('ring group (0) min usage cannot be satisfied due to fragment constraint'.
                             format(base_symbol))
                self.enough_ring = False
            elif rrange.max > total_count:
                logger.info('ring group {0} max usage changed from {1} to {2} due to fragment constraint'.
                            format(base_symbol, rrange.max, total_count))
                rrange.max = total_count

    def check_feasibility_of_aromatic_fragments(self, generator):
        """Check feasibility of ring resource for aromatic fragment

        Args:
            generator (ChemGraphGenerator): structure generator

        Returns:
            bool: true if feasible
        """
        # check availability of rings and graphs for aromatic only fragments
        for fragment, frange in self.fragment_const.items():
            if frange.min == 0:
                continue
            # check if aromatic fragment
            aromatic_fragment = True
            for v in fragment.graph.vertices:
                if v.aromatic_atom():
                    if any([e.bond_type != BondType.AROMATIC for e in v.edges]):
                        aromatic_fragment = False
                        break
                else:
                    aromatic_fragment = False
                    break
            if not aromatic_fragment:
                continue
            # check occurrence of aromatic fragment in rings
            found_graph = False
            for ring_symbol, rrange in self.ring_resource.items():
                ring_resource = generator.ring_res_mgr.get_resource(ring_symbol)
                ring_graph = ring_resource.get_vertex()
                ring_label = ring_resource.get_labeling()
                if rrange.max > 0 and fragment.count_fragment_graph(ring_graph, ring_label) > 0:
                    found_graph = True
                    break
            if found_graph:
                continue
            logger.info('no ring/graph resource for aromatic fragment: {0}:{1}'.
                  format(generator.fragment_smiles[fragment], frange))
            return False
        return True

    def print_constraint(self, generator):
        """print the contents of the constraint of structure generation

        Args:
            generator (ChemGraphGenerator): structure generator
        """
        print('structure generation constraint: {0}'.format(self.id))
        if self.total_atom_range is not None:
            print('total atom:{0}'.format(self.total_atom_range))
        for atom, num_range in self.atom_resource.items():
            print('atom:{0} \'{1}\' valence={2}'.
                  format(num_range, atom, generator.atom_res_mgr.get_valence(atom)))
        if self.total_ring_range is not None:
            print('total ring:{0}'.format(self.total_ring_range))
        for ring, num_range in self.ring_range.items():
            print('ring_{0}:{1}'.format(ring, num_range))
        if self.total_aromatic_range is not None:
            print('total aromatic ring:{0}'.format(self.total_aromatic_range))
        for ring, num_range in self.aromatic_range.items():
            print('aromatic_{0}:{1}'.format(ring, num_range))
        for base_symbol, num_range in self.ring_group_resource.items():
            base_resource = generator.ring_res_mgr.get_base_resource(base_symbol)
            print('ring group:{0} \'{1}\' members={2}'.
                  format(num_range, base_symbol, len(base_resource.get_members())))
        for ring_symbol, num_range in self.ring_resource.items():
            print('ring:{0} \'{1}\''.
                  format(num_range, ring_symbol))
        for f in sorted(self.fragment_const.keys(), key=lambda g: g.graph.to_smiles()):
            print('fragment:{0} \'{1}\' {2}'.
                  format(self.fragment_const[f], generator.fragment_smiles[f], f.to_string()))
        for features, f_list in self.online_features.items():
            print('online features:{0} {1}'.format(features.get_id(), [f.get_id() for f in f_list]))
        for (atom0, bond_type), atoms in sorted(self.prohibited_connections1.items()):
            bond_char = ChemGraph.get_bond_char(bond_type)
            bond_char = '-' if bond_char == '' else bond_char
            atoms_str = '{'
            for atom1 in sorted(atoms):
                atoms_str += atom1 + ','
            atoms_str = atoms_str.rstrip(',') + '}'
            print('prohibited atom:[{0}]{1} atoms:{2}'.format(atom0, bond_char, atoms_str))
        for (atom0, bond_type0), edges in sorted(self.prohibited_connections2.items()):
            bond_char0 = ChemGraph.get_bond_char(bond_type0)
            bond_char0 = '-' if bond_char0 == '' else bond_char0
            edges_str = '{'
            for (atom2, bond_type1, atom1) in sorted(edges):
                bond_char1 = ChemGraph.get_bond_char(bond_type1)
                bond_char1 = '-' if bond_char1 == '' else bond_char1
                edges_str += '[' + atom2 + ']' + bond_char1 + '[' + atom1 + ']' + ','
            edges_str = edges_str.rstrip(',') + '}'
            print('prohibited atom:[{0}]{1} edges:{2}'.format(atom0, bond_char0, edges_str))

    def update_available_atom_resource(self, generator, node, remaining_atom, available_atom):
        """Update available atoms by checking the number of used atoms.

        Args:
            generator (ChemGraphGenerator): structure generator
            node (AtomGraphNode): current search tree node
            remaining_atom (Counter): a list of remaining atoms
            available_atom (list): a list of available atoms

        Returns:
            list: a list of updated available atoms
        """
        if self.total_atom_range is not None:
            if node.total_atom_count >= self.total_atom_range.max:
                return []
        # check each atom count
        new_available_atom = []
        for atom in available_atom:
            if remaining_atom[atom] > 0:
                new_available_atom.append(atom)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('new available_atom={0}'.format(new_available_atom))
        return new_available_atom

    def update_available_ring_resource(self, generator, node, remaining_atom, available_ring):
        """Update available rings by checking the number of used atoms.

        Args:
            generator (ChemGraphGenerator): structure generator
            node (AtomGraphNode): current search tree node
            remaining_atom (Counter): a dictionary of remaining atoms
            available_ring (list): a list of available rings

        Returns:
            list: a list of updated available rings
        """
        # get feasible ring count
        if self.total_ring_range is not None:
            if node.total_ring_count >= self.total_ring_range.max:
                return []
        # get available ring for extension
        new_available_ring = []
        for ring_symbol in available_ring:
            if ring_symbol not in self.ring_resource:
                continue
            ring_resource = generator.ring_res_mgr.get_resource(ring_symbol)
            ring_vertex = ring_resource.get_vertex()
            base_symbol = ring_resource.get_base_symbol()
            if node.ring_count[ring_symbol] >= self.ring_resource[ring_symbol].max:
                continue
            if base_symbol is not None and \
                    node.ring_group_count[base_symbol] >= self.ring_group_resource[base_symbol].max:
                continue
            # check atom availability
            ring_extension = ring_resource.get_extension()
            if self.check_graph_counter_range(node, ring_vertex) and \
                    all(remaining_atom[atom] >= count for atom, count in ring_resource.atom_count.items()) and \
                    (node.parent is None or len(ring_extension) > 0):
                new_available_ring.append(ring_symbol)
        if logger.isEnabledFor(logging.INFO):
            logger.info('new available_ring={0}'.format(new_available_ring))
        return new_available_ring

    def check_graph_counter_range(self, node, graph):
        """Check if adding a new sub-graph as a vertex is feasible or not.

        Args:
            node (AtomGraphNode): current search tree node
            graph (SubStructureAsVertex): a new graph adding as a vertex

        Returns:
            bool: True if feasible. False otherwise
        """
        if self.total_atom_range is not None:
            if node.total_atom_count + sum(graph.atom_count.values()) > self.total_atom_range.max:
                return False
        for ring, rrange in self.ring_range.items():
            if node.num_ring_count[ring] + graph.ring_count[ring] > rrange.max:
                return False
        if self.total_ring_range is not None:
            if node.total_ring_count + sum(graph.ring_count.values()) > self.total_ring_range.max:
                return False
        for aromatic, arange in self.aromatic_range.items():
            if node.num_aromatic_count[aromatic] + graph.aromatic_count[aromatic] > arange.max:
                return False
        if self.total_aromatic_range is not None:
            if node.total_aromatic_count + sum(graph.aromatic_count.values()) > self.total_aromatic_range.max:
                return False
        return True

    def check_terminate_ring(self, generator, node, available_ring, remaining_atom):
        """Check the termination by rings.

        Args:
            generator (ChemGraphGenerator): structure generator
            node (AtomGraphNode): current search tree node
            available_ring (list): a list of available rings
            remaining_atom (Counter): a dictionary of remaining atoms

        Returns:
            bool: termination of the node
        """
        # get total number of available atoms
        total_available_all = self.total_atom_count
        if self.total_atom_range is not None:
            total_available_all = min(total_available_all, self.total_atom_range.max)
        total_available_all -= node.total_atom_count
        # check atom availability for base rings
        available_ring_set = set(available_ring)
        total_required_atom = Counter()
        total_required_all = 0
        for base_symbol, num_range in self.ring_group_resource.items():
            base_resource = generator.ring_res_mgr.get_base_resource(base_symbol)
            if node.ring_group_count[base_symbol] < num_range.min:
                ring_group_atoms = base_resource.get_group_atom_count()
                ring_group_atom_all = base_resource.get_vertex().num_atom()
                required_ring = num_range.min - node.ring_group_count[base_symbol]
                if all([m.get_symbol() not in available_ring_set for m in base_resource.get_members()]):
                    reason = 'required base ring is not available:{0}'.format(base_symbol)
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(reason)
                    return True, reason
                # keep available atom for a ring group before updating remaining atoms
                group_available_all = sum([remaining_atom[atom] for atom in ring_group_atoms])
                for atom, count in ring_group_atoms.items():
                    required_atom = required_ring * count
                    total_required_atom[atom] += required_atom
                    if required_atom > remaining_atom[atom]:
                        reason = 'atom require for base ring:{0} atom {1}'.format(base_symbol, atom)
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
                    # update remaining atom
                    remaining_atom[atom] -= required_atom
                if ring_group_atom_all * required_ring > group_available_all:
                    reason = 'atom require for base ring:{0} atom total'.format(base_symbol)
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(reason)
                    return True, reason
                elif ring_group_atom_all * required_ring == group_available_all:
                    # update remaining atom
                    for atom in ring_group_atoms:
                        remaining_atom[atom] = 0
                # check total atoms
                total_required_all += ring_group_atom_all * required_ring
                if total_required_all > total_available_all:
                    reason = 'atom required for ring:atom total'
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(reason)
                    return True, reason
                elif total_required_all == total_available_all:
                    # update remaining atom
                    remaining_atom.clear()
        # check atom availability of non-base rings
        for ring_symbol, num_range in self.ring_resource.items():
            ring_resource = generator.ring_res_mgr.get_resource(ring_symbol)
            if ring_resource.get_base_resource() is None and \
                    node.ring_count[ring_symbol] < num_range.min:
                ring_atoms = ring_resource.get_vertex().atom_count
                required_ring = num_range.min - node.ring_count[ring_symbol]
                # keep available atom for a ring before updating remaining atoms
                for atom, count in ring_atoms.items():
                    required_atom = required_ring * count
                    total_required_atom[atom] += required_atom
                    if required_atom > remaining_atom[atom]:
                        reason = 'atom require for ring:{0} atom {1}'.format(ring_symbol, atom)
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
                    # update remaining atom
                    remaining_atom[atom] -= required_atom
        return False, ''

    def check_terminate_counter_range(self, generator, node, available_ring, remaining_atom):
        """Check the termination by estimating lower bound of required atoms to satisfy remaining ring ranges
        by available ring resources.

        Args:
            generator (ChemGraphGenerator): structure generator
            node (AtomGraphNode): current search tree node
            available_ring (list): a list of available rings
            remaining_atom (Counter): a dictionary of remaining atoms

        Returns:
            bool: termination of the node
        """
        # get required rings and aromatic rings of each size
        total_required_ring = 0
        total_required_aromatic = 0
        required_ring = Counter()
        required_aromatic = Counter()
        if self.total_ring_range is not None:
            total_required_ring = max(0, self.total_ring_range.min - node.total_ring_count)
        if self.total_aromatic_range is not None:
            total_required_aromatic = max(0, self.total_aromatic_range.min - node.total_aromatic_count)
        for ring, rrange in self.ring_range.items():
            required = rrange.min - node.num_ring_count[ring]
            if required > 0:
                required_ring[ring] = required
        for aromatic, arange in self.aromatic_range.items():
            required = arange.min - node.num_aromatic_count[aromatic]
            if required > 0:
                required_aromatic[aromatic] = required
        need_to_check_ring = (total_required_ring > 0 or len(required_ring) > 0)
        need_to_check_aromatic = (total_required_aromatic > 0 or len(required_aromatic) > 0)

        if not need_to_check_ring and not need_to_check_aromatic:
            return False, ''

        # get total number of available atoms
        total_available_all = self.total_atom_count
        if self.total_atom_range is not None:
            total_available_all = min(total_available_all, self.total_atom_range.max)
        total_available_all -= node.total_atom_count

        # estimate lower bond of required atoms to satisfy ring counts
        total_ring_required_atom = 0
        total_aromatic_required_atom = 0
        ring_required_atom = Counter()
        aromatic_required_atom = Counter()
        for ring_symbol in available_ring:
            ring_resource = generator.ring_res_mgr.get_resource(ring_symbol)
            ring_graph = ring_resource.get_vertex()
            if ring_graph.total_ring_count > 0 and need_to_check_ring:
                # total required atoms
                required_num = total_required_ring / ring_graph.total_ring_count
                total_ring_required_atom = required_num * ring_graph.num_atom() if total_ring_required_atom == 0 \
                    else min(total_ring_required_atom, required_num * ring_graph.num_atom())
                # required atoms for each atom symbol
                required_num = max(required_ring.values()) if len(required_ring) > 0 else 0
                for ring, count in ring_graph.ring_count.items():
                    required_num = min(required_num, required_ring[ring] / count)
                self.update_required_atoms(required_num, ring_resource.atom_count, ring_required_atom)
            if ring_graph.total_aromatic_count > 0 and need_to_check_aromatic:
                # total required atoms
                required_num = total_required_aromatic / ring_graph.total_aromatic_count
                total_aromatic_required_atom = required_num * ring_graph.num_atom() if total_aromatic_required_atom == 0 \
                    else min(total_aromatic_required_atom, required_num * ring_graph.num_atom())
                # required atoms for each atom symbol
                required_num = max(required_aromatic.values()) if len(required_aromatic) > 0 else 0
                for aromatic, count in ring_graph.aromatic_count.items():
                    required_num = min(required_num, required_aromatic[aromatic] / count)
                self.update_required_atoms(required_num, ring_resource.atom_count, aromatic_required_atom)
        # check available atom
        if need_to_check_ring:
            if total_required_ring > 0 and total_ring_required_atom == 0:
                reason = 'no ring is available for rings'
                if logger.isEnabledFor(logging.INFO):
                    logger.info(reason)
                return True, reason
            for atom, count in ring_required_atom.items():
                if count > remaining_atom[atom]:
                    reason = 'less required atom for ring:{0} {1}'.format(atom, count)
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(reason)
                    return True, reason
                # update remaining atom
                remaining_atom[atom] -= count
            if total_ring_required_atom > total_available_all:
                reason = 'less required total atom for rings'
                if logger.isEnabledFor(logging.INFO):
                    logger.info(reason)
                return True, reason
            elif total_ring_required_atom == total_available_all:
                # update remaining atom
                remaining_atom.clear()
        elif need_to_check_aromatic:
            if total_required_aromatic > 0 and total_aromatic_required_atom == 0:
                reason = 'no ring is availabel for aromatic rings'
                if logger.isEnabledFor(logging.INFO):
                    logger.info(reason)
                return True, reason
            for atom, count in aromatic_required_atom.items():
                if count > remaining_atom[atom]:
                    reason = 'less required atom for aromatic ring:{0} {1}'.format(atom, count)
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(reason)
                    return True, reason
                # update remaining atom
                remaining_atom[atom] -= count
            if total_aromatic_required_atom > total_available_all:
                reason = 'less required total atom for aromatic rings'
                if logger.isEnabledFor(logging.INFO):
                    logger.info(reason)
                return True, reason
            elif total_aromatic_required_atom == total_available_all:
                # update remaining atom
                remaining_atom.clear()
        return False, ''

    def update_required_atoms(self, required_num, atom_count, ring_required_atom):
        """Update the number of required atoms to use a ring

        Args:
            required_num: number of required number of rings
            atom_count: required atom count for rings
            ring_required_atom: required atom to use a ring
        """
        for atom in self.atom_resource:
            required_atom = atom_count[atom] * required_num
            if atom in ring_required_atom:
                ring_required_atom[atom] = min(ring_required_atom[atom], required_atom)
            else:
                ring_required_atom[atom] = required_atom
        return

    def check_terminate_fragment(self, generator, node):
        """Check the continuation by fragment.

        Args:
            generator (ChemGraphGenerator): structure generator
            node (AtomGraphNode): current search tree node

        Returns:
            bool: termination of the node
        """
        # get total number of available atoms
        total_available_all = self.total_atom_count
        if self.total_atom_range is not None:
            total_available_all = min(total_available_all, self.total_atom_range.max)
        total_available_all -= node.total_atom_count
        # check fragment satisfaction
        for fragment, num_range in self.fragment_const.items():
            node_fragment_count = node.fragment_count[fragment]
            if not fragment.exact_match:
                if fragment.root_vertex is None:
                    # sub-structure
                    if node_fragment_count > num_range.max:
                        reason = 'required fragment exceed:count={0} {1}'.\
                            format(node_fragment_count, fragment.to_string())
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
                elif fragment.root_vertex.atom in node.atom_degree_count:
                    # fp-structure
                    degree_count = node.atom_degree_count[fragment.root_vertex.atom]
                    bond_degree = fragment.root_vertex.bond_degree()
                    if sum(degree_count[0:bond_degree]) + node_fragment_count < num_range.min:
                        reason = 'required fragment short:count={0} {1}'.\
                            format(node_fragment_count, fragment.to_string())
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
                    if bond_degree + 1 == len(degree_count) and node_fragment_count > num_range.max:
                        reason = 'required fragment exceed:count={0} {1}'.\
                            format(node_fragment_count, fragment.to_string())
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
                    if node_fragment_count - total_available_all > num_range.max:
                        reason = 'required fragment exceed:count={0} {1}'.\
                            format(node_fragment_count, fragment.to_string())
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
            # if fragment include rings check available atom for rings
            if num_range.min > 0 and node_fragment_count == 0 and fragment.graph.num_ring_atom() > 0:
                n_aromatic_atom = node.graph.num_aromatic_atom()
                f_aromatic_atom = fragment.graph.num_aromatic_atom()
                n_plain_ring_atom = node.graph.num_ring_atom()-n_aromatic_atom
                f_plain_ring_atom = fragment.graph.num_ring_atom()-f_aromatic_atom
                if n_aromatic_atom < f_aromatic_atom:
                    if total_available_all < f_aromatic_atom - n_aromatic_atom:
                        reason = 'aromatic atom require for fragment:node={0} f={1} {2}'.\
                            format(n_aromatic_atom, f_aromatic_atom, generator.fragment_smiles[fragment])
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
                if n_plain_ring_atom < f_plain_ring_atom:
                    if total_available_all < f_plain_ring_atom - n_plain_ring_atom:
                        reason = 'ring atom require for fragment:node={0} f={1} {2}'.\
                            format(n_plain_ring_atom, f_plain_ring_atom, generator.fragment_smiles[fragment])
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(reason)
                        return True, reason
        return False, ''

    def check_counter_ranges(self, node):
        """Check the ranges of atom, ring, aromatic rings.

        Args:
            node (AtomGraphNode): current search tree node

        Returns:
            bool: satisfiability of the node
        """
        if self.total_atom_range is not None:
            if not self.total_atom_range.contains(node.total_atom_count):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('total atom short/over:count=%d' % node.total_atom_count)
                return False
        for ring, rrange in self.ring_range.items():
            if not rrange.contains(node.num_ring_count[ring]):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('num ring short/over:%d count=%d' % (ring, node.num_ring_count[ring]))
                return False
        if self.total_ring_range is not None:
            if not self.total_ring_range.contains(node.total_ring_count):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('total ring short/over:count=%d' % node.total_ring_count)
                return False
        for aromatic, arange in self.aromatic_range.items():
            if not arange.contains(node.num_aromatic_count[aromatic]):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('num aromatic short/over:%d count=%d' % (aromatic, node.num_aromatic_count[aromatic]))
                return False
        if self.total_aromatic_range is not None:
            if not self.total_aromatic_range.contains(node.total_aromatic_count):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('total aromatic short/over:count=%d' % node.total_aromatic_count)
                return False
        return True

    def check_atom_ranges(self, node):
        """Check the feasibility of the number of atoms.

        Args:
            node (AtomGraphNode): current search tree node

        Returns:
            bool: satisfiability of the node
        """
        # check atom resource satisfaction
        for atom, arange in self.atom_resource.items():
            if not arange.contains(node.atom_count[atom]):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('atom short/over:count=%d %s', node.atom_count[atom], atom)
                return False
        return True

    def check_ring_ranges(self, node):
        """Check the feasibility of the number of rings.

        Args:
            node (AtomGraphNode): current search tree node

        Returns:
            bool: satisfiability of the node
        """
        # check base ring resource satisfaction
        for base_symbol, num_range in self.ring_group_resource.items():
            if not num_range.contains(node.ring_group_count[base_symbol]):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('base ring short/over:count={0} {1}'.
                                format(node.ring_group_count[base_symbol], base_symbol))
                return False
        # check ring resource satisfaction
        for ring_symbol, num_range in self.ring_resource.items():
            if not num_range.contains(node.ring_count[ring_symbol]):
                if logger.isEnabledFor(logging.INFO):
                    logger.info('ring short/over:count={0} {1}'.
                                format(node.ring_group_count[ring_symbol], ring_symbol))
                return False
        return True

    def check_fragment_ranges(self, node):
        """Check the feasibility of the number of fragment occurrences and the continuation.

        Args:
            node (AtomGraphNode): current search tree node

        Returns:
            bool, bool: satisfiability of the node, continuation of the search
        """
        # check fragment satisfaction
        for fragment, num_range in self.fragment_const.items():
            node_fragment_count = node.fragment_count[fragment]
            if node_fragment_count > num_range.max:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('fragment exceed:count=%d %s', node_fragment_count, fragment.to_string())
                return False
            if node_fragment_count < num_range.min:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('fragment short:count=%d %s', node_fragment_count, fragment.to_string())
                return False

        return True

    def score_search_node(self, node, available_atom, available_ring):
        """Get score of the search node.

        Args:
            node (AtomGraphNode): current search tree node
            available_atom (list): a list of available atoms
            available_ring (list): a list of available rings

        Returns:
            float: score of satisfiability of the node
        """
        if self.infeasible:
            return self.invalid_node_score

        atom_weight = 1.0
        ring_weight = 1.0
        fragment_weight = 2.0

        score = 0
        num_test = 0
        # check atom/ring/aring count range
        if self.total_atom_range is not None:
            num_test += 1 * atom_weight
            score += self.calc_score(node.total_atom_count, self.total_atom_range, atom_weight)

        for ring, rrange in self.ring_range.items():
            num_test += 1 * ring_weight
            score += self.calc_score(node.num_ring_count[ring], rrange, ring_weight)

        if self.total_ring_range is not None:
            num_test += 1 * ring_weight
            score += self.calc_score(node.total_ring_count, self.total_ring_range, ring_weight)

        for aromatic, arange in self.aromatic_range.items():
            num_test += 1 * ring_weight
            score += self.calc_score(node.num_aromatic_count[aromatic], arange, ring_weight)

        if self.total_aromatic_range is not None:
            num_test += 1 * ring_weight
            score += self.calc_score(node.total_aromatic_count, self.total_aromatic_range, ring_weight)

        # check atom resource satisfaction
        # for atom_symbol in available_atom:
        for atom_symbol, num_range in self.atom_resource.items():
            num_test += 1 * atom_weight
            score += self.calc_score(node.atom_count[atom_symbol], num_range, atom_weight)

        # check base ring resource satisfaction
        for base_symbol, num_range in self.ring_group_resource.items():
            num_test += 1 * ring_weight
            score += self.calc_score(node.ring_group_count[base_symbol], num_range, ring_weight)

        # check ring resource satisfaction
        for ring_symbol, num_range in self.ring_resource.items():
            num_test += 1 * ring_weight
            score += self.calc_score(node.ring_count[ring_symbol], num_range, ring_weight)

        # check fragment satisfaction
        for fragment, num_range in self.fragment_const.items():
            num_test += 1 * fragment_weight
            node_fragment_count = node.fragment_count[fragment]
            if fragment.should_mark_vertex():
                # fp-structure
                score += self.calc_fp_score(node_fragment_count, num_range, fragment_weight)
            else:
                # sub-structure
                score += self.calc_score(node_fragment_count, num_range, fragment_weight)

        if num_test > 0:
            return (score / num_test) * node.total_atom_count
        else:
            return score

    def score_non_feature_search_node(self, node, available_atom, available_ring):
        """Get score of the search node only considering features not in the feature vector

        Args:
            node (AtomGraphNode): current search tree node
            available_atom (list): a list of available atoms
            available_ring (list): a list of available rings

        Returns:
            float: score of satisfiability of the node
        """
        if self.infeasible:
            return self.invalid_node_score

        fragment_weight = 2.0

        score = 0
        num_test = 0

        # check fragment satisfaction
        for fragment, num_range in self.fragment_const.items():
            if fragment in self.fragments_in_feature_vector:
                continue
            num_test += 1 * fragment_weight
            node_fragment_count = node.fragment_count[fragment]
            if fragment.should_mark_vertex():
                # fp-structure
                score += self.calc_fp_score(node_fragment_count, num_range, fragment_weight)
            else:
                # sub-structure
                score += self.calc_score(node_fragment_count, num_range, fragment_weight)

        if num_test > 0:
            return (score / num_test) * node.total_atom_count
        else:
            return score

    @staticmethod
    def calc_score(count, num_range, weight):
        if count <= num_range.min:
            return (num_range.min - count) / (num_range.min + 1) * weight
        elif count <= num_range.max:
            return (count - num_range.min) / (num_range.max + 1) * weight
        else:
            return 0

    @staticmethod
    def calc_fp_score(count, num_range, weight):
        if count <= num_range.min:
            return (num_range.min - count) / (num_range.min + 1) * weight
        elif num_range.max <= count:
            return (count - num_range.max) / (num_range.max + 1) * weight
        else:
            return 0

    def check_prohibited_connection_of_graph(self, graph):
        """Check prohibited connections in a graph

        Args:
            graph (AtomGraph): a graph

        Returns:
            bool: true if permitted graph
        """
        prohibited = False
        for v in graph.vertices:
            v.visit = 1
            for e_index in range(len(v.edges)):
                e = v.edges[e_index]
                if e.end.visit == 0 and \
                        not self.check_prohibited_connection1(e.start, e.end.atom, e.bond_type):
                    prohibited = True
                    break
                for e0_index in range(e_index + 1, len(v.edges)):
                    e0 = v.edges[e0_index]
                    if not self.check_prohibited_connection2(e0, e.end.atom, e.bond_type):
                        prohibited = True
                        break
                if prohibited:
                    break
            if prohibited:
                break
        for v in graph.vertices:
            v.visit = 0
        return not prohibited

    def check_prohibited_connection1(self, vertex, atom, bond_type):
        """Check the atom connection by prohibited connection list

        Args:
            vertex (AtomVertex): connecting vertex
            atom (str): start atom symbol
            bond_type (BondType): bond type of start atom connection

        Returns:
            bool: true if permitted connection
        """
        if atom <= vertex.atom:
            if vertex.atom in self.prohibited_connections1.get((atom, bond_type), {}):
                return False
        else:
            if atom in self.prohibited_connections1.get((vertex.atom, bond_type), {}):
                return False
        return True

    def check_prohibited_connection2(self, edge, atom, bond_type):
        """Check the edge connection by prohibited connection list

        Args:
            edge (AtomEdge): connecting edge
            atom (str): start atom symbol
            bond_type (BondType): bond type of start atom connection

        Returns:
            bool: true if permitted connection
        """
        atom0 = atom
        atom1 = edge.end.atom
        atom2 = edge.start.atom
        bond_type0 = bond_type
        bond_type1 = edge.bond_type
        if atom0 == atom1:
            edge0 = (atom0, min(bond_type0, bond_type1))
            edge1 = (atom2, max(bond_type0, bond_type1), atom1)
        elif atom0 <= atom1:
            edge0 = (atom0, bond_type0)
            edge1 = (atom2, bond_type1, atom1)
        else:
            edge0 = (atom1, bond_type1)
            edge1 = (atom2, bond_type0, atom0)

        if edge1 in self.prohibited_connections2.get(edge0, {}):
            return False
        else:
            return True


class ChemGraphConstraintSet(ChemGraphConstraint):
    """A set of constraints of the structure generation as ranges of atoms, rings, fragments.

    Attributes:
        constraints (list): a list of ChemGraphConstraint
        prediction_error (float): acceptable range of prediction error
        molecule_evaluator (MoleculeEvaluator): an evaluator of feature vector of a molecule
    """

    def __init__(self, constraints, prediction_error, molecule_evaluator=None):
        """Constructor of a set of molecular graph constraints.

        Args:
            constraints (list): a list of ChemGraphConstraint objects
            prediction_error (float): acceptable range of prediction error
            molecule_evaluator (MoleculeEvaluator): molecule feature vector evaluator
        """
        # get resource constraint as a union of resource ranges
        atom_constraint = {}
        ring_constraint = {}
        fragment_constraint = {}
        online_features = {}
        for constraint in constraints:
            for atom_def, arange in constraint.atom_constraint.items():
                if atom_def in atom_constraint:
                    old_range = atom_constraint[atom_def]
                    atom_constraint[atom_def] = [min(old_range[0], arange[0]), max(old_range[1], arange[1])]
                else:
                    atom_constraint[atom_def] = arange
            for ring_def, rrange in constraint.ring_constraint.items():
                if ring_def in ring_constraint:
                    old_range = ring_constraint[ring_def]
                    ring_constraint[ring_def] = [min(old_range[0], rrange[0]), max(old_range[1], rrange[1])]
                else:
                    ring_constraint[ring_def] = rrange
            for fragment_def, frange in constraint.fragment_constraint.items():
                if fragment_def in fragment_constraint:
                    old_range = fragment_constraint[fragment_def]
                    fragment_constraint[fragment_def] = [min(old_range[0], frange[0]), max(old_range[1], frange[1])]
                else:
                    fragment_constraint[fragment_def] = frange
            online_features = constraint.online_features

        super().__init__('constraint set({0})'.format(len(constraints)),
                         atom_constraint, ring_constraint, fragment_constraint,
                         online_features,
                         prediction_error=prediction_error, molecule_evaluator=molecule_evaluator)

        self.constraints = constraints
        self.infeasible = []

        for constraint in self.constraints:
            if constraint.ring_range is not None:
                for rnum, rrange in constraint.ring_range.items():
                    if rnum in self.ring_range:
                        self.ring_range[rnum] = rrange.union(self.ring_range[rnum])
                    else:
                        self.ring_range[rnum] = rrange
            if constraint.aromatic_range is not None:
                for rnum, rrange in constraint.aromatic_range.items():
                    if rnum in self.aromatic_range:
                        self.aromatic_range[rnum] = rrange.union(self.aromatic_range[rnum])
                    else:
                        self.aromatic_range[rnum] = rrange
            if constraint.total_atom_range is not None:
                if self.total_atom_range is not None:
                    self.total_atom_range = constraint.total_atom_range.union(self.total_atom_range)
                else:
                    self.total_atom_range = constraint.total_atom_range
            if constraint.total_ring_range is not None:
                if self.total_ring_range is not None:
                    self.total_ring_range = constraint.total_ring_range.union(self.total_ring_range)
                else:
                    self.total_ring_range = constraint.total_ring_range
            if constraint.total_aromatic_range is not None:
                if self.total_aromatic_range is not None:
                    self.total_aromatic_range = constraint.total_aromatic_range.union(self.total_aromatic_range)
                else:
                    self.total_aromatic_range = constraint.total_aromatic_range

    def set_resource_constraint(self, generator):
        """Initialize constraints for the resources

        Args:
            generator (ChemGraphGenerator): structure generator
        """
        for constraint in self.constraints:
            constraint.set_resource_constraint(generator)

        super().set_resource_constraint(generator)

    def set_infeasible(self, constraints):
        """Add infeasible constraint

        Args:
            a list of infeasible constraints
        """
        self.infeasible.extend(constraints)

    def reset_infeasible(self, constraints):
        """Remove infeasible constraint

        Args:
            a list of infeasible constraint to remove
        """
        for constraint in reversed(constraints):
            constraint.reset_infeasible(constraint)
            self.infeasible.pop()

    def is_infeasible(self):
        """Check if all the constraints are infeasible

        Returns:
            bool: true if all the constraints are infeasible
        """
        return len(self.infeasible) == len(self.constraints)

    def get_reason(self):
        """Get a reason of infeasibility

        Returns:
            str: reason of infeasibility
        """
        return self.infeasible[-1].infeasible_reason

    def check_terminate(self, generator, node, available_atom, available_ring):
        """Check the termination of search node by considering remaining atoms and unsatisfied constraints

        Args:
            generator (ChemGraphGenerator): the structure generator
            node (AtomGraphNode): search node of the generator
            available_atom (list): a list of available atoms
            available_ring (list): a list of available rings

        Returns:
            list, list, list, list: a list of infeasible constraint object, list of atoms, list of rings,
                list of candidate atoms for next search
        """
        new_infeasible = []
        new_available_atom = set()
        new_available_ring = set()
        atom_candidate = set()
        for constraint in self.constraints:
            if constraint.is_infeasible():
                continue
            infeasible, c_available_atom, c_available_ring, c_atom_candidate = \
                constraint.check_terminate(generator, node, available_atom, available_ring)
            if infeasible is not None:
                new_infeasible.append(constraint)
                continue
            for atom in c_available_atom:
                new_available_atom.add(atom)
            for ring in c_available_ring:
                new_available_ring.add(ring)
            for atom in c_atom_candidate:
                atom_candidate.add(atom)

        self.set_infeasible(new_infeasible)
        new_available_atom = sorted(list(new_available_atom), reverse=True)
        new_available_ring = sorted(list(new_available_ring), reverse=True)
        atom_candidate = sorted(list(atom_candidate), reverse=True)
        return new_infeasible, new_available_atom, new_available_ring, atom_candidate

    def check_resource_satisfaction(self, node):
        """Check if some constraints are satisfied at a search node

        Args:
            node (AtomGraphNode): a search node

        Returns:
            set: a set of constraints
        """
        satisfied_constraints = set()
        # check satisfaction by feature vectors
        for constraint in self.constraints:
            satisfied_constraint = constraint.check_resource_satisfaction(node)
            if len(satisfied_constraint) > 0:
                satisfied_constraints.add(constraint.get_id())
        # check satisfaction by regression function
        if len(satisfied_constraints) == 0:
            if self.molecule_evaluator is not None:
                # all the values are determined by molecular structure at a search node
                property_estimates = self.molecule_evaluator.evaluate_molecule(node, verify=False)
                if property_estimates is not None:
                    max_score = 0
                    for _, (estimate, score) in property_estimates.items():
                        max_score = max(max_score, score)
                    if max_score < self.prediction_error ** 2:
                        if self.check_non_feature_satisfaction(node):
                            satisfied_constraints.add('')
            else:
                # some feature vector values are referred to individual feature vectors
                local_vector = None
                for constraint in self.constraints:
                    if constraint.molecule_evaluator is None:
                        continue
                    if local_vector is None:
                        local_vector = constraint.molecule_evaluator.\
                            get_local_feature_vector(node, verify=False)
                    property_estimates = constraint.molecule_evaluator.\
                        evaluate_molecule(node, local_vector=local_vector, verify=False)
                    if property_estimates is not None:
                        max_score = 0
                        for _, (estimate, score) in property_estimates.items():
                            max_score = max(max_score, score)
                        if max_score < self.prediction_error ** 2:
                            if constraint.check_non_feature_satisfaction(node):
                                satisfied_constraints.add(constraint.get_id())
        return satisfied_constraints

    def check_non_feature_satisfaction(self, node):
        """Check if some constraints not related to feature vector are satisfied

        Args:
            node (AtomGraphNode): a search node

        Returns:
            bool: true if satisfied
        """
        for constraint in self.constraints:
            if constraint.check_non_feature_satisfaction(node):
                return True
        return False

    def score_search_node(self, node, available_atom, available_ring):
        """Get score of the search node.

        Args:
            node (AtomGraphNode): current search tree node
            available_atom (list): a list of available atoms
            available_ring (list): a list of available rings

        Returns:
            float: score of satisfiability of the node
        """
        score = 0
        for constraint in self.constraints:
            const_score = constraint.score_search_node(node, available_atom, available_ring)
            if score == 0:
                score = const_score
            else:
                score = min(score, const_score)
        return score

    def print_constraint(self, generator):
        """print the contents of the constraint of structure generation

        Args:
            generator (ChemGraphGenerator): structure generator
        """
        super().print_constraint(generator)
        for constraint in self.constraints:
            constraint.print_constraint(generator)


class ChemGraphConstraintByRegressionFunction(ChemGraphConstraint):
    """Constraints of the structure generation only by a regression function.

    Attributes:
        A_ub (matrix): coefficient matrix of linear programming
        b_ub (array): constants of linear programming
        bounds (list): bounds of column variable of linear programming
        resource_index (tuple): a tuple of dictionaries of resource name and column index
        fragment_feature (dict): a mapping of fragment object and feature name
        check_terminate_with_lin_prog (tool): flag of checking feasibility by linear programming
    """

    def __init__(self, id, atom_constraint, ring_constraint, fragment_constraint,
                 online_features,
                 ring_range=None, aromatic_range=None,
                 total_atom_range=None, total_ring_range=None, total_aromatic_range=None, 
                 prediction_error=1.0, molecule_evaluator=None):
        """Constructor of molecular graph constraint.

        Args:
            id (str): constraint id
            atom_constraint (dict): a dictionary of feasible ranges of atom definitions
            ring_constraint (dict): a dictionary of feasible ranges of ring definitions
            fragment_constraint (dict): a dictionary of feasible ranges of fragment definitions
            online_features (dict): a dictionary of online update feature sets
            aromatic_range (dict, optional): feasible range of the number of aromatic rings. 
                Defaults to None.
            total_atom_range (list, optional): feasible range of the number of total atoms. 
                Defaults to None.
            total_ring_range(list, optional): feasible range of the number of total rings.
                Defaults to None.
            total_aromatic_range(list, optional): feasible range of the number of total aromatic rings. 
                Defaults to None.
        """
        super().__init__(id, atom_constraint, ring_constraint, fragment_constraint,
                         online_features,
                         ring_range, aromatic_range,
                         total_atom_range, total_ring_range, total_aromatic_range, 
                         prediction_error, molecule_evaluator)

        if molecule_evaluator is None:
            raise ValueError('MoleculeEvaluator is required')

        self.A_ub = None
        self.b_ub = None
        self.bounds = None
        self.resource_index = None
        self.fragment_feature = None
        self.check_range_with_lin_prog = False
        self.check_terminate_with_lin_prog = False

    def set_resource_constraint(self, generator):
        """Initialize constraints for the resources

        Args:
            generator (ChemGraphGenerator): structure generator
        """
        super().set_resource_constraint(generator)

        # update resource ranges by linear programming
        if self.check_range_with_lin_prog and self.molecule_evaluator.is_linear_model():
            self.A_ub, self.b_ub, self.resource_index, self.fragment_feature = self.make_lin_matrix()
            self.bounds = self.update_resource_range(self.A_ub, self.b_ub, self.resource_index, self.fragment_feature)
        #     self.check_by_fragment_constraint(generator)

    def check_terminate(self, generator, node, available_atom, available_ring):
        """Check the termination of search node by considering remaining atoms and unsatisfied constraints

        Args:
            generator (ChemGraphGenerator): the structure generator
            node (AtomGraphNode): search node of the generator
            available_atom (list): a list of available atoms
            available_ring (list): a list of available rings

        Returns:
            ChemGraphConstraint, list, list, list: infeasible constraint object, list of atoms, list of rings,
                list of candidate atoms for next search
        """
        infeasible, new_available_atom, new_available_ring, candidate_atom = \
            super().check_terminate(generator, node, available_atom, available_ring)

        if not self.check_terminate_with_lin_prog:
            return infeasible, new_available_atom, new_available_ring, candidate_atom

        # if feasible, check infeasibility by linear programming
        if infeasible is None:
            # check feasible region by linear programming
            atom_index, ring_index, aromatic_index, sub_index, fp_index, f_index = self.resource_index
            for atom, index in atom_index.items():
                self.bounds[index].min = node.atom_count[atom]
            for ring, index in ring_index.items():
                self.bounds[index].min = node.ring_count[ring]
            for aromatic, index in aromatic_index.items():
                self.bounds[index].min = node.num_aromatic_count[aromatic]
            for fragment, index in sub_index.items():
                self.bounds[index].min = node.fragment_count[fragment]
            for fragment, index in fp_index.items():
                self.bounds[index].min = node.fragment_count[fragment]
            bounds = [(b.min, b.max) for b in self.bounds]
            if not self.check_feasible_range(self.A_ub, self.b_ub, bounds, False):
                logger.info('terminate by lin prog:node={0}'.format(generator.node_count))
                infeasible = self

        return infeasible, new_available_atom, new_available_ring, candidate_atom

    def check_feasible_range(self, A_ub, b_ub, bounds, integral_point):
        """Check if solution space is feasible or not

        Args:
            A_ub (matrix): coefficient matrix of linear programming
            b_ub (array): constants of linear programming
            bounds (list): list of original bounds of column variables
            integral_point (bool): flag for checking feasibility by integral point in the solution space

        Returns:
            bool: if solution space is feasible or not
        """
        column_size = A_ub.shape[1]

        if not integral_point:
            c = np.zeros(column_size)
            c[0] = 1
            opt_result = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
            if opt_result.success:
                return True
            else:
                return False
        else:
            for index in range(column_size):
                c = np.zeros(column_size)
                c[index] = 1
                new_bound = NumRange(0)
                opt_result = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex')
                if opt_result.success:
                    raw_value = opt_result.x[index]
                    new_bound.min = int(math.ceil(raw_value - sys.float_info.epsilon * raw_value))
                else:
                    return False
                opt_result = sp.optimize.linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex')
                if opt_result.success:
                    raw_value = opt_result.x[index]
                    new_bound.max = int(math.floor(raw_value + sys.float_info.epsilon * raw_value))
                else:
                    return False
                # check existence of integral point
                if new_bound.min > new_bound.max:
                    return False
            return True

    def check_resource_satisfaction(self, node):
        """Check if a constraint is satisfied at a search node

        Args:
            node (AtomGraphNode): a search node

        Returns:
            set: a set of constraints
        """
        satisfied_constraints = super().check_resource_satisfaction(node)
        if len(satisfied_constraints) > 0:
            property_estimates = self.molecule_evaluator.evaluate_molecule(node, verify=False)
            if property_estimates is None:
                return dict()
            max_score = 0
            for _, (estimate, score) in property_estimates.items():
                max_score = max(max_score, score)
            if max_score < self.prediction_error ** 2:
                return satisfied_constraints
        return dict()

    def score_search_node(self, node, available_atom, available_ring):
        """Get score of the search node.

        Args:
            node (AtomGraphNode): current search tree node
            available_atom (list): a list of available atoms
            available_ring (list): a list of available rings

        Returns:
            float: score of satisfiability of the node
        """
        property_estimates = self.molecule_evaluator.evaluate_molecule(node)
        if property_estimates is None:
            return self.invalid_node_score
        max_score = 0
        for _, (estimate, score) in property_estimates.items():
            max_score = max(max_score, score)
        score = math.sqrt(max_score) * node.total_atom_count
        score += self.score_non_feature_search_node(node, available_atom, available_ring)
        return score

    def make_lin_matrix(self):
        """Make a matrix for linear programming of solution space

        Returns:
            matrix, array, tuple, dict: coefs of lin prog, const of lin prog, resource index, feature names
        """
        # get coefficients of regression model
        coefs, consts, targets, errors = self.molecule_evaluator.get_coefs
        f_atom_index, f_ring_index, f_aromatic_index, f_sub_fragment_index, f_fp_fragment_index, feature_index = \
            self.molecule_evaluator.get_resource_index()
        feature_vector_size = len(coefs[0])

        # make a mapping of fragment to fragment feature id
        fragment_feature = dict()
        for feature in f_sub_fragment_index:
            fragment_feature[feature.get_fragment()] = feature.get_id()
        for feature in f_fp_fragment_index:
            fragment_feature[feature.get_fragment()] = feature.get_id()

        # consider all the resources (not in the feature vectors)
        column_size = feature_vector_size
        # define column index of atoms
        atom_index = dict()
        for feature, index in f_atom_index.items():
            atom_index[feature.get_index()] = index
        for atom in self.atom_resource:
            if atom not in atom_index:
                atom_index[atom] = column_size
                column_size += 1
        # define column index of rings
        ring_index = dict()
        for feature, index in f_ring_index.items():
            ring_index[feature.get_index()] = index
        for ring_size in self.ring_range:
            if ring_size not in ring_index:
                ring_index[ring_size] = column_size
                column_size += 1
        # define column index of aromatic rings
        aromatic_index = dict()
        for feature, index in f_aromatic_index.items():
            aromatic_index[feature.get_index()] = index
        for ring_size in self.aromatic_range:
            if ring_size not in aromatic_index:
                aromatic_index[ring_size] = column_size
                column_size += 1
        # define column index of sub_structure fragments
        sub_fragment_index = dict()
        for feature, index in f_sub_fragment_index.items():
            sub_fragment_index[feature.get_index()] = index
        for fragment in self.fragment_const:
            if not fragment.should_mark_vertex():
                if fragment not in sub_fragment_index:
                    sub_fragment_index[fragment] = column_size
                    column_size += 1
        # define column index of fp_structure fragments
        fp_fragment_index = dict()
        for feature, index in f_fp_fragment_index.items():
            fp_fragment_index[feature.get_index()] = index
        for fragment in self.fragment_const:
            if fragment.should_mark_vertex():
                if fragment not in fp_fragment_index:
                    fp_fragment_index[fragment] = column_size
                    column_size += 1

        # define matrix for linear programming
        A_row = []
        b_row = []
        # set constraint of regression model
        for index in range(len(coefs)):
            row = np.zeros(column_size)
            row[:feature_vector_size] = coefs[index]
            # set upper bound of regression value
            if targets[index][1] <= sys.float_info.max:
                A_row.append(row)
                b_row.append(consts[index] + targets[index][1] + self.prediction_error * errors[index])
            # set lower bound of regression value
            if targets[index][0] >= -sys.float_info.max:
                A_row.append(-row)
                b_row.append(-consts[index] - targets[index][0] + self.prediction_error * errors[index])
        # set constraint of total atom
        row = np.zeros(column_size)
        for atom, index in atom_index.items():
            row[index] = 1
        A_row.append(row)
        b_row.append(self.total_atom_range.max)
        A_row.append(-row)
        b_row.append(-self.total_atom_range.min)
        # set constraint of total ring
        row = np.zeros(column_size)
        for ring, index in ring_index.items():
            row[index] = 1
        A_row.append(row)
        b_row.append(self.total_ring_range.max)
        A_row.append(-row)
        b_row.append(-self.total_ring_range.min)
        # set constraint of total aromatic ring
        row = np.zeros(column_size)
        for ring, index in aromatic_index.items():
            row[index] = 1
        A_row.append(row)
        b_row.append(self.total_aromatic_range.max)
        A_row.append(-row)
        b_row.append(-self.total_aromatic_range.min)

        # make np matrix
        A_ub = np.stack(A_row)
        b_ub = np.array(b_row)
        resource_index = (atom_index, ring_index, aromatic_index, sub_fragment_index, fp_fragment_index, feature_index)

        return A_ub, b_ub, resource_index, fragment_feature

    def update_resource_range(self, A_ub, b_ub, resource_index, fragment_feature):
        """Update ranges of resources by soling linear programming

        Args:
            A_ub (matrix): coefficient matrix of linear programming
            b_ub (array): constants of linear programming
            resource_index (tuple): a tuple of dictionaries of resource names and thier column index
            fragment_feature (dict): a mapping of fragment object and feature name

        Returns:
            list: a list of revised bound of resources
        """

        column_size = A_ub.shape[1]
        atom_index, ring_index, aromatic_index, sub_fragment_index, fp_fragment_index, feature_index = resource_index

        # define bound of variables
        bounds = [(None, None) for _ in range(column_size)]
        for atom, index in atom_index.items():
            rrange = self.atom_resource[atom]
            bounds[index] = (rrange.min, rrange.max)
        for ring, index in ring_index.items():
            rrange = self.ring_range[ring]
            bounds[index] = (rrange.min, rrange.max)
        for aromatic, index in aromatic_index.items():
            rrange = self.aromatic_range[aromatic]
            bounds[index] = (rrange.min, rrange.max)
        for fragment, index in sub_fragment_index.items():
            rrange = self.fragment_const[fragment]
            bounds[index] = (rrange.min, rrange.max)
        for fragment, index in fp_fragment_index.items():
            rrange = self.fragment_const[fragment]
            bounds[index] = (rrange.min, rrange.max)

        # get new range of resources by solving linear program
        revised_bounds = []
        revised_raw_bounds = []
        for index in range(column_size):
            new_raw_bound = [0, 0]
            new_bound = NumRange(0)
            if index not in feature_index.values():
                c = np.zeros(column_size)
                c[index] = 1
                opt_result = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex')
                if opt_result.success:
                    raw_value = opt_result.x[index]
                    new_raw_bound[0] = raw_value
                    new_bound.min = int(math.ceil(raw_value - sys.float_info.epsilon))
                else:
                    logger.warning('resource range is infeasible')
                    break
                opt_result = sp.optimize.linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex')
                if opt_result.success:
                    raw_value = opt_result.x[index]
                    new_raw_bound[1] = raw_value
                    new_bound.max = int(math.floor(raw_value + sys.float_info.epsilon))
                else:
                    logger.warning('resource range is infeasible')
                    break
                # check existence of integral point
                if new_bound.min > new_bound.max:
                    logger.warning('no integral point: index={0}'.format(index))
                    break
            revised_raw_bounds.append(new_raw_bound)
            revised_bounds.append(new_bound)

        if len(revised_bounds) < column_size:
            revised_bounds = [NumRange(0)] * column_size

        # update range of resources
        rstr = ''
        for atom, index in atom_index.items():
            bound = self.atom_resource[atom]
            new_bound = revised_bounds[index]
            if not (bound.min == new_bound.min and bound.max == new_bound.max):
                self.atom_resource[atom] = new_bound
                rstr += '{0}:{1}->{2} '.format(atom, bound, new_bound)
        if rstr != '':
            logger.info('revised atom:{0}'.format(rstr))
        rstr = ''
        for ring, index in ring_index.items():
            bound = self.ring_range[ring]
            new_bound = revised_bounds[index]
            if not (bound.min == new_bound.min and bound.max == new_bound.max):
                self.ring_range[ring] = new_bound
                rstr += '{0}:{1}->{2} '.format(ring, bound, new_bound)
        if rstr != '':
            logger.info('revised ring:{0}'.format(rstr))
        rstr = ''
        for aromatic, index in aromatic_index.items():
            bound = self.aromatic_range[aromatic]
            new_bound = revised_bounds[index]
            if not (bound.min == new_bound.min and bound.max == new_bound.max):
                self.aromatic_range[aromatic] = new_bound
                rstr += '{0}:{1}->{2} '.format(aromatic, bound, new_bound)
        if rstr != '':
            logger.info('revised aromatic:{0}'.format(rstr))
        rstr = ''
        for fragment, index in sub_fragment_index.items():
            bound = self.fragment_const[fragment]
            new_bound = revised_bounds[index]
            if not (bound.min == new_bound.min and bound.max == new_bound.max):
                self.fragment_const[fragment] = new_bound
                if fragment in fragment_feature:
                    rstr += '{0}:{1}->{2} '.format(fragment_feature[fragment], bound, new_bound)
                else:
                    rstr += '{0}:{1}->{2} '.format(fragment.graph.to_smiles(), bound, new_bound)
        if rstr != '':
            logger.info('revised sub structure:{0}'.format(rstr))
        rstr = ''
        for fragment, index in fp_fragment_index.items():
            bound = self.fragment_const[fragment]
            new_bound = revised_bounds[index]
            if not (bound.min == new_bound.min and bound.max == new_bound.max):
                self.fragment_const[fragment] = new_bound
                if fragment in fragment_feature:
                    rstr += '{0}:{1}->{2} '.format(fragment_feature[fragment], bound, new_bound)
                else:
                    rstr += '{0}:{1}->{2} '.format(fragment.graph.to_smiles(), bound, new_bound)
        if rstr != '':
            logger.info('revised fp structure:{0}'.format(rstr))

        return revised_bounds


class ChemGraphGenerator(object):
    """Driver of molecular graph generation. Remaining available resources for extending a graph
    (atoms, rings, and sub-structures) are checked, and only feasible search tree node is extended
    by adding new atom, ring, and sub-structure. Generated molecular graphs are stored in a member
    variable as a list of SMILES.

    Attributes:
        constraint (ChemGraphConstraint): constraint of atom, ring and fragment resources
        atom_res_mgr (AtomResManager): a resource manager of atom resource
        ring_res_mgr (RingResManager): a resource manager of ring resource
        fragment_object (dict): a dictionary of fragment definition and reference
        fragment_template (dict): a dictonary of gragment reference and fragment object
        fragment_smiles (dict): a dictionary of SMILES representation of fragments
        solution (defaultdict): a dictionary of constraint id and list of generated graphs
        solution_count (int): number of generated graphs
        node_count (int): number of generated search tree nodes
        depth_count (dict): a mapping of depth and the number of node
        depth_score (list): a list of score of nodes in the leafs
        depth_threshold (dict): a mapping of depth and the threshold for pruning
        max_solution (int): maximum number of solutions to generate
        max_node (int): maximum number of search tree nodes to generate
        max_depth (int): maximum depth of search tree
        iterative_depth (int): maximum depth of each iterative search
        beam_size (int): width of beam size
        max_bond (int): maximum bond to add a new atom
        active_node (dict): a dictionary of active search nodes in beam search
        verbose (bool): flag of verbose mode in the generation
    """

    progress_interval = 10000
    """The number of search nodes for reporting progress"""

    bond_type_list = [
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
    ]
    """List of bond types used in the generation"""

    def __init__(self, resource_constraint, verbose=True):
        """Constructor of molecular graph generator.

        Args:
            resource_constraint (ChemGraphConstraint): constraint of atom, ring and fragment resources
            verbose (bool): flag of verbose mode
        """
        self.atom_res_mgr = AtomResManager()
        self.ring_res_mgr = RingResManager()
        self.fragment_object = {}
        self.fragment_template = {}
        self.fragment_smiles = {}
        self.solution = defaultdict(list)
        self.solution_count = 0
        self.node_count = 0
        self.depth_count = Counter()
        self.depth_score = []
        self.depth_threshold = {}
        self.start_graph = None
        self.max_solution = 0
        self.max_node = 0
        self.max_depth = 0
        self.iterative_depth = 0
        self.beam_size = 0
        self.max_bond = 0
        self.iterative_depth = 0
        self.active_node = dict()
        self.verbose = verbose

        # set constraint object
        self.constraint = resource_constraint

        # create atom, ring, graph and fragment resource objects
        atom_defs, ring_defs, fragment_defs = self.constraint.get_resource_definitions()

        self.atom_res_mgr.create_resources(atom_defs)
        self.ring_res_mgr.create_resources(ring_defs, self.atom_res_mgr)
        self.create_constraint_fragment(fragment_defs)

        # initialize constraint object
        self.constraint.set_resource_constraint(self)

    def create_constraint_fragment(self, fragment_definitions):
        """create fragment object from fragment definition
        fragment definition is
        (1) smiles of fragment
        (2) AtomGraph object
        (3) ChemFragment object

        Args:
            fragment_definitions (set): a set of fragment definitions
        """
        for fragment_def in fragment_definitions:
            if isinstance(fragment_def, str):
                fragment_sm = ChemGraph.canonical_smiles(fragment_def)
                if fragment_sm is None:
                    logger.error('invalid smiles %s for fragment: ignored', fragment_def)
                    continue
                fragment = ChemFragment(AtomGraph(smarts=fragment_def))
            elif isinstance(fragment_def, AtomGraph):
                fragment = ChemFragment(fragment_def)
            elif isinstance(fragment_def, ChemFragment):
                fragment = fragment_def
            else:
                logger.error('invalid fragment definition %s', fragment_def)
                continue
            # update valence of fragment graph
            self.atom_res_mgr.update_valence(fragment.graph)
            # set fragment constraint
            self.fragment_object[fragment_def] = fragment
            self.fragment_template[fragment] = fragment
            self.fragment_smiles[fragment] = fragment.graph.to_smiles()

    def get_fragment_object(self, fragment_definition):
        """get fragment object from fragment_definition

        Args:
            fragment_definition (str, AtomGraph, ChemFragment): fragment definition

        Returns:
            ChemFragment: fragment object
        """
        if fragment_definition in self.fragment_object:
            return self.fragment_object[fragment_definition]
        else:
            logger.error('no fragment object is created for %s', fragment_definition)
            return None

    def get_solution(self):
        """Get a dictionary of generated graphs

        Returns:
            dict: a dictionary of graphs
        """
        return self.solution

    def get_num_solution(self):
        """Get the number of unique solutions

        Returns:
            int: the number of solutions
        """
        return self.solution_count

    def search(self, max_solution=0, max_node=0, max_bond=3, max_depth=0, beam_size=0): 
        """Search for feasible molecular graphs corresponding to specified ranges of resources and
        fragment constraints by depth first tree search. Found graphs are stored in solution.

        Args:
            max_solution (int, optional): maximum number of solutions to find. Defaults to 0.
            max_node (int, optional): maximum number of search tree nodes to search. Defaults to 0.
            max_bond (int, optional): maximum number of bonds in the generation. Defaults to 3.
            max_depth (int, optional): maximum depth of search tree. Defaults to 0.
            beam_size (int, optional): width of beam search. Defaults to 0.

        Returns:
            bool: true if normal termination
        """
        # start chemical graph generation
        if self.verbose:
            self.constraint.print_constraint(self)

        self.solution = defaultdict(list)
        self.solution_count = 0
        self.node_count = 0
        self.depth_count = Counter()
        self.depth_score = []
        self.depth_threshold = dict()
        self.max_solution = max_solution
        self.max_node = max_node
        self.max_depth = max_depth
        self.iterative_depth = 0
        self.active_node = dict()
        self.beam_size = beam_size
        self.max_bond = max_bond

        if not self.constraint.enough_atom:
            logger.info('there is not enough atoms for satisfying fragment constraints')
            logger.info('%d molecules are found (%d nodes)' % (self.solution_count, self.node_count))
            return True

        if not self.constraint.enough_ring:
            logger.info('minimum number of rings are not satisfied by fragment constraints')
            logger.info('%d molecules are found (%d nodes)' % (self.solution_count, self.node_count))
            return True

        if not self.constraint.check_feasibility_of_aromatic_fragments(self):
            logger.info('%d molecules are found (%d nodes)' % (self.solution_count, self.node_count))
            return True

        # start from an empty root search tree node
        root_graph = AtomGraph()
        root_labeling = ChemGraphLabeling(root_graph.vertices)
        root_node = AtomGraphNode(None, root_graph, None, root_labeling)
        root_node.set_online_features(self.constraint.get_online_features())
        root_node.set_fragment_list(self.constraint.get_fragments_to_count())

        # set atom degree counter to root_node
        for atom_symbol, nrange in self.constraint.atom_resource.items():
            if nrange.max > 0:
                num_valence = self.atom_res_mgr.get_valence(atom_symbol)
                degree_count = np.zeros(num_valence + 1)
                degree_count[0] = nrange.max
                root_node.atom_degree_count[atom_symbol] = degree_count

        # initialize available atom/ring/graph
        available_atom = sorted(self.atom_res_mgr.get_symbols(), reverse=True)
        available_ring = sorted(list(self.ring_res_mgr.get_symbols()), reverse=True)

        # check termination
        reason, available_atom, available_ring, atom_candidate = \
            self.constraint.check_terminate(self, root_node, available_atom, available_ring)

        if self.constraint.is_infeasible():
            logger.info('{0} molecules are found ({1} nodes): {2}'.
                  format(self.solution_count, self.node_count, self.constraint.get_reason()))
            self.constraint.reset_infeasible(reason)
            return True

        # get must-be-root atom from fp-structure
        atom_candidate = self.constraint.update_root_atom_candidate(atom_candidate)

        # loop of iterative depth
        self.solution = defaultdict(list)
        self.solution_count = 0
        self.depth_threshold = {}
        interrupt = False
        max_depth_count = Counter()
        while not self.terminate_check():
            # initialization of each iterative search
            self.iterative_depth += 1
            self.node_count = 0
            self.depth_count = Counter()
            self.depth_score = []

            if len(available_ring) > 0:
                try:
                    self.generation_by_ring(root_node, None, None,
                                            available_ring, available_atom)
                except KeyboardInterrupt:
                    logger.info('stop by key interruption')
                    interrupt = True
                    break
                # check max_node
                if self.terminate_check():
                    break

            if len(atom_candidate) > 0:
                try:
                    self.generation_by_atom(root_node, None, None, atom_candidate,
                                            available_ring, available_atom)
                except KeyboardInterrupt:
                    logger.info('stop by key interruption')
                    interrupt = True
                    break
                # check max_node
                if self.terminate_check():
                    break

            # set depth threshold
            if self.beam_size > 0:
                if 0 < self.beam_size < len(self.depth_score):
                    self.depth_threshold[self.iterative_depth] = self.depth_score[self.beam_size-1]
                elif len(self.depth_score) > 0:
                    self.depth_score.sort()
                    self.depth_threshold[self.iterative_depth] = self.depth_score[-1]

            # print solutions and nodes for this iteration
            if self.beam_size == 0:
                logger.info('{0} solutions {1} nodes depth={2} (depth_node={3}) {4}'.
                      format(self.solution_count, self.node_count,
                             self.max_depth,
                             list(self.depth_count.values()), self.start_graph))
            else:
                max_depth_count[self.iterative_depth] = self.depth_count[self.iterative_depth]
                logger.info('{0} solutions {1}/{2} nodes depth={3} (depth_node={4}) {5}'.
                      format(self.solution_count, self.node_count, sum(max_depth_count.values()),
                             self.iterative_depth,
                             list(max_depth_count.values()), self.start_graph))

            if self.beam_size > 0:
                # check leaf node, return before end depth
                if self.depth_count[self.iterative_depth] == 0:
                    break
                # check max depth, reach max_depth
                if self.max_depth in self.depth_count:
                    break
                # check atom range
                if self.constraint.total_atom_range is not None and \
                        self.constraint.total_atom_range.max == self.iterative_depth:
                    break
            else:
                # no iteration for zero beam size
                break

        # clear cache of resources
        self.atom_res_mgr.clear_cache()
        self.ring_res_mgr.clear_cache()

        if self.beam_size == 0:
            logger.info('{0} molecules are found ({1} nodes)'.
                  format(self.solution_count, self.node_count))
        else:
            max_depth_count[self.iterative_depth] = self.depth_count[self.iterative_depth]
            logger.info('{0} molecules are found ({1}/{2} nodes)'.
                  format(self.solution_count, self.node_count, sum(max_depth_count.values())))

        return False if interrupt else True

    def search0(self, node, available_atom, available_ring):
        """Search for a feasible molecular graph from a given search tree node. The search tree node
        is extended by adding available atom, ring, and sub-structures.

        Args:
            node (AtomGraphNode): current seach tree node
            available_atom (list): a list of available atoms to extend a graph
            available_ring (list): a list of available rings to extend a graph
        """

        if logger.isEnabledFor(logging.INFO):
            logger.info('search:{0} {1}'.format(node.get_generation_path(),
                                                node.graph.to_string()))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('atom_count:%s', node.atom_count)
            logger.debug('ring_count:%s', node.ring_count)

        # check atom,ring,fragment resource satisfaction
        resource_satisfy = set()
        if self.beam_size == 0 or node.depth == self.iterative_depth:
            resource_satisfy = self.constraint.check_resource_satisfaction(node)

        # check singleton of ring
        # only the first instance of single ring is acceptable as a solution
        if len(node.graph.vertices) == 1:
            if len(node.graph.ring_vertices) > 0:
                v = node.graph.ring_vertices[0]
                e = node.resource.get_extension()
                if len(e) > 0 and v.connection_vertex.connect.ring_index != list(e.keys())[0]:
                    resource_satisfy = set()

        if len(resource_satisfy) > 0:
            # resources are in expected range
            # expand shrunk ring graph vertex
            node.graph.expand_graph()
            self.solution_count += 1
            for candidate_id in resource_satisfy:
                new_graph = copy.deepcopy(node.graph)
                path = node.get_op_sequence()
                solution = (new_graph, path)
                self.solution[candidate_id].append(solution)
            # print('%s'%(smiles))
            if self.verbose:
                print('const:{0}'.format(list(resource_satisfy)))
                print('%d/%d %s %s' % (self.solution_count, self.node_count,
                                       node.graph.to_smiles(), node.graph.to_string()))
            if logger.isEnabledFor(logging.INFO):
                logger.info('found:%s', node.graph.to_string())
            # shrink ring graph vertex
            node.graph.shrink_graph()

        if (self.verbose or self.beam_size == 0) and self.node_count % self.progress_interval == 0:
            print('{0} solutions {1} nodes depth={2} (depth_node={3}) {4}'.
                  format(self.solution_count, self.node_count,
                         self.max_depth if self.beam_size == 0 else self.iterative_depth,
                         list(self.depth_count.values()), self.start_graph))

        # check terminate condition
        if self.terminate_check():
            return 0

        # check termination
        infeasible, available_atom, available_ring, atom_candidate = \
            self.constraint.check_terminate(self, node, available_atom, available_ring)

        # check node pruning for beam search
        terminate = False
        reason = ''
        if self.beam_size > 0:
            if self.constraint.is_infeasible():
                node_score = self.constraint.invalid_node_score
            else:
                node_score = self.constraint.score_search_node(node, available_atom, available_ring)
            if node.depth < self.iterative_depth:
                # old depth in iterative deepening
                if node_score > self.depth_threshold[node.depth]:
                    reason = 'beam size threshold'
                    terminate = True
            else:
                # new depth in iterative deepening
                self.depth_score.append(node_score)
                if len(self.depth_score) > self.beam_size:
                    self.depth_score.sort()
                    if node_score > self.depth_score[self.beam_size]:
                        reason = 'beam size threshold'
                        if logger.isEnabledFor(logging.INFO):
                            logger.info('terminate search node ({0}): {1}'.format(node.depth, reason))
                        self.constraint.reset_infeasible(infeasible)
                        return 0
                self.active_node[node.get_generation_path()] = node.automorphism
                reason = 'reach iterative depth'
                if logger.isEnabledFor(logging.INFO):
                    logger.info('terminate search node ({0}): {1}'.format(node.depth, reason))
                self.constraint.reset_infeasible(infeasible)
                return 1

        # check max depth
        if 0 < self.max_depth <= node.depth:
            reason = 'reach max depth'
            terminate = True

        if terminate or self.constraint.is_infeasible():
            if reason == '':
                reason = self.constraint.get_reason()
            if logger.isEnabledFor(logging.INFO):
                logger.info('terminate search node ({0}): {1}'.format(node.depth, reason))
            if self.beam_size > 0:
                del self.active_node[node.get_generation_path()]
            self.constraint.reset_infeasible(infeasible)
            return 0

        active_offspring_count = 0
        if len(available_atom) > 0:
            # extend node with new atom
            # get extending positions of current graph
            position_to_extend = node.position_to_extend()
            if logger.isEnabledFor(logging.INFO):
                logger.info('position_to_extend=%s', [v.index for v in position_to_extend])
            # create a new node of extended graph with an available atom
            for position in position_to_extend:
                # if position is connecting vertex, expand a graph/ring graph
                # and find real position to extend in graph/ring vertices
                graph_vertex = position
                if isinstance(position, RingConnectionVertex):
                    graph_vertex0 = node.graph.get_ring_graph_vertex(position)
                    e = self.ring_res_mgr.get_resource(graph_vertex0.symbol).get_extension()
                    if len(node.graph.vertices) > 1:
                        graph_vertex = graph_vertex0
                        real_positions = [graph_vertex0.vertices[index] for index in e[position.connect.ring_index][0]]
                    else:
                        real_positions = [position]
                        graph_vertex0.automorphism = e[position.connect.ring_index][1]
                else:
                    real_positions = [position]

                # expand graph vertex
                node.expand_graph_vertex(graph_vertex)
                if logger.isEnabledFor(logging.INFO):
                    if isinstance(position, RingConnectionVertex):
                        logger.info('ring expand:<[%d],%s>', position.connect.ring_index, node.graph.to_string())

                # extend graph from real position to extend
                for real_position in real_positions:
                    max_bond = int(real_position.num_free_hand())
                    max_bond = min(max_bond, self.max_bond)
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('try position:{0}:{1} free bond:{2}'.
                                    format(real_position.index, real_position.atom, max_bond))
                    for bond_type in self.bond_type_list:
                        # check bond order
                        if ChemGraph.BOND_ORDER[bond_type] > max_bond:
                            continue
                        # check bond availability for graph vertex
                        if not real_position.bond_available(bond_type):
                            continue
                        if len(available_ring) > 0:
                            # generation by adding ring
                            active_offspring_count += \
                                self.generation_by_ring(node, real_position, bond_type,
                                                        available_ring, available_atom)
                            # check max_node
                            if self.terminate_check():
                                break
                        if len(atom_candidate) > 0:
                            active_offspring_count += \
                                self.generation_by_atom(node, real_position, bond_type, atom_candidate,
                                                        available_ring, available_atom)
                            # check max_node
                            if self.terminate_check():
                                break
                    if self.terminate_check():
                        break
                # shrink expanded ring graphs
                node.shrink_graph_vertex()

                if self.terminate_check():
                    break

        # update active node
        if self.beam_size > 0 and active_offspring_count == 0:
            del self.active_node[node.get_generation_path()]

        self.constraint.reset_infeasible(infeasible)
        return active_offspring_count

    def generation_by_ring(self, node, position, bond_type, available_ring, available_atom):
        """Extend a node by adding a new ring.

        Args:
            node (AtomGraphNode): current search tree node
            position (AtomVertex): an atom vertex to add a new ring
            bond_type (BondType: bond type of an edge connecting a new vertex
            available_ring (list): a list of available rings to extend a graph
            available_atom (list): a list of available atoms to extend a graph
        """
        # try generation by ring
        active_offspring_count = 0
        for ring_symbol in available_ring:
            ring_resource = self.ring_res_mgr.get_resource(ring_symbol)
            ring_vertex = ring_resource.get_vertex()
            ring_extension = ring_resource.get_extension()
            if position is None and len(ring_extension) == 0:
                # even if a ring does not have extension points,
                # whole molecule is added to root node
                ring_extension = [0]
            for index in ring_extension:
                # check atom connection
                if not self.check_atom_connection(position, ring_vertex.vertices[index], bond_type):
                    if logger.isEnabledFor(logging.INFO):
                        if position is None:
                            logger.info('no root ring:{0} connect:<[{1}],{2}>'.
                                        format(ring_symbol, index, node.graph.to_string()))
                        else:
                            logger.info('no ring:{0} connect:<[{1},{2},b={3}],{4}>'.
                                        format(ring_symbol, position.index, index, bond_type, node.graph.to_string()))
                    continue
                if self.beam_size > 0 and node.depth + 1 < self.iterative_depth:
                    new_node = node.extend_by_ring(position, ring_resource, index, bond_type,
                                                   self.active_node, True)
                else:
                    new_node = node.extend_by_ring(position, ring_resource, index, bond_type,
                                                   None, True)
                if new_node is not None:
                    self.node_count += 1
                    self.depth_count[new_node.depth] += 1
                    if position is None:
                        self.start_graph = ring_symbol
                    if logger.isEnabledFor(logging.INFO):
                        if position is None:
                            logger.info('root ring:{0} extend:<[{1}],{2}>'.format(ring_symbol, index,
                                                                                  new_node.graph.to_string()))
                        else:
                            logger.info('ring:{0} extend:<[{1},{2},b={3}],{4}>'.format(ring_symbol, position.index,
                                                                                       index, bond_type,
                                                                                       new_node.graph.to_string()))
                    active_offspring_count += \
                        self.search0(new_node, available_atom, available_ring)
                    new_node.restore()
                else:
                    if logger.isEnabledFor(logging.INFO):
                        if position is None:
                            logger.info('no root ring:{0} extend:<[{1}],{2}'.
                                        format(ring_symbol, index, node.graph.to_string()))
                        else:
                            logger.info('no ring:{0} extend:<[{1},{2},b={3}],{4}'.
                                        format(ring_symbol, position.index, index, bond_type,
                                               node.graph.to_string()))
                # check max_node
                if self.terminate_check():
                    return active_offspring_count

        return active_offspring_count

    def generation_by_atom(self, node, position, bond_type, atom_candidate, available_ring, available_atom):
        """Extend a node by adding a new atom.

        Args:
            node (AtomGraphNode): current search tree node
            position (AtomVertex): an atom vertex to add a new atom
            bond_type (BondType): bond type of an edge connecting a new vertex
            atom_candidate (list): a list of candidate atoms for search
            available_ring (list): a list of available rings to extend a graph
            available_atom (list): a list of available atoms to extend a graph
        """
        # try generation by atom
        active_offspring_count = 0
        for atom in atom_candidate:
            atom_resource = self.atom_res_mgr.get_resource(atom)
            atom_vertex = atom_resource.get_vertex()
            if not self.check_atom_connection(position, atom_vertex, bond_type):
                if logger.isEnabledFor(logging.INFO):
                    if position is None:
                        logger.info('no root atom:%s connect:<[],%s>', atom, node.graph.to_string())
                    else:
                        logger.info('no atom:%s connect:<[%d,b=%s],%s>', atom, position.index, bond_type,
                                    node.graph.to_string())
                continue
            if self.beam_size > 0 and node.depth + 1 < self.iterative_depth:
                new_node = node.extend_by_atom(position, atom_resource, bond_type,
                                               self.active_node, True)
            else:
                new_node = node.extend_by_atom(position, atom_resource, bond_type,
                                               None, True)
            if new_node is not None:
                self.node_count += 1
                self.depth_count[new_node.depth] += 1
                if position is None:
                    self.start_graph = atom_vertex.atom
                if logger.isEnabledFor(logging.INFO):
                    if position is None:
                        logger.info('root atom:%s extend:<[],%s>', atom, new_node.graph.to_string())
                    else:
                        logger.info('atom:%s extend:<[%d],%s>', atom, position.index, new_node.graph.to_string())
                active_offspring_count += \
                    self.search0(new_node, available_atom, available_ring)
                new_node.restore()
            else:
                if logger.isEnabledFor(logging.INFO):
                    if position is None:
                        logger.info('no root atom:%s extend:<[],%s>', atom, node.graph.to_string())
                    else:
                        logger.info('no atom:%s extend:<[%d,b=%s],%s>', atom, position.index, bond_type,
                                    node.graph.to_string())
            # check max node
            if self.terminate_check():
                return active_offspring_count
        return active_offspring_count

    def check_atom_connection(self, vertex, vertex_extend, bond_type):
        """Check the feasibility to add vertex_extend to vertex with a given bond type

        Args:
            vertex (AtomVertex): vertex to add a new vertex
            vertex_extend (AtomVertex, str): a new vertex
            bond_type (BondType): bond type of atom connection
        """
        if vertex is None:
            return True

        # check free valence
        if not vertex_extend.bond_available(bond_type):
            return False

        vertex = vertex.get_connect_vertex()
        vertex_extend = vertex_extend.get_connect_vertex()

        # check prohibited connection1
        if len(self.constraint.prohibited_connections1) > 0:
            if not self.constraint.check_prohibited_connection1(vertex, vertex_extend.atom, bond_type):
                return False

        # check prohibited connection2
        if len(self.constraint.prohibited_connections2) > 0:
            for edge in vertex.edges:
                if not self.constraint.check_prohibited_connection2(edge, vertex_extend.atom, bond_type):
                    return False

        return True

    def terminate_check(self):
        """Check the termination of the search by referring to maximum number of solutions and nodes.

        Returns:
            bool: True if maximum number of solutions or nodes is reached. False otherwise.
        """
        if 0 < self.max_node <= self.node_count:
            return True
        if 0 < self.max_solution <= self.solution_count:
            return True
        return False
