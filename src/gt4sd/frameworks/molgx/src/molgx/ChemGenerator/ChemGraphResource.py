# -*- coding:utf-8 -*-
"""
ChemGraphResource.py

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
from .ChemRingGen import *

from collections import Counter
import copy

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GraphGenResourceManager(object):
    """Base class of factory and manager of atom/ring/graph resources in the structure generation

    Attributes:
        symbol_res_map (dict): a dictionary from symbol to resource
        def_symbol_map (dict): a dictionary from resource definition to symbol
    """

    class Resource(object):
        """Base class of a resource (atom/ring) of the structure generation

        Attributes:
            symbol (str): symbol of a resource
            vertex (AtomVertex): a vertex object
            instance (AtomVertex): an instance of a vertex object
            atom_count (Counter): number of atoms in a resource
            ring_count (Counter): number of rings in a resource
            aromatic_count (Counter): number of aromatic ring in a resource
            fragment_count (Counter): number of fragments in a resource

        """
        def __init__(self, symbol, vertex):
            """Constructor of GraphGenResourceManager.Resource

            Args:
                symbol (str): symbol of a resource
                vertex (AtomVertex): a graph vertex object
            """
            self.symbol = symbol
            self.vertex = vertex
            self.instance = None
            self.atom_count = Counter()
            self.ring_count = Counter()
            self.aromatic_count = Counter()
            self.fragment_count = Counter()
            self.fragment_path = list()
            self.vertex_cache = list()
            self.vertex_in_use = list()

        def get_symbol(self):
            """Get a symbol of a resource

            Returns:
                str: symbol of a resource
            """
            return self.symbol

        def get_vertex(self):
            """Get a vertex object

            Returns:
                AtomVertex: a vertex object
            """
            return self.vertex

        def get_instance(self):
            """Get current instance of resource vertex

            Returns:
                AtomVertex: a vertex instance
            """
            return self.instance

        def new_instance(self):
            """Get a new instance of resource vertex

            Returns:
                AtomVertex: a vertex instance
            """
            # keep current instance
            self.vertex_in_use.append(self.instance)
            # get new instance
            if len(self.vertex_cache) > 0:
                # get new instance from cache
                self.instance = self.vertex_cache.pop()
            else:
                if isinstance(self.vertex, SubStructureAsVertex):
                    # save exact_match_fragments
                    match_fragment_dict = dict()
                    for v in self.vertex.vertices:
                        match_fragment_dict[v] = v.exact_match_fragment
                        v.exact_match_fragment = set()
                # create a new instance by copying a vertex
                self.instance = copy.deepcopy(self.vertex)
                if isinstance(self.vertex, SubStructureAsVertex):
                    # restore exact_match_fragment
                    vertex_map = {ov: iv for (ov, iv) in zip(self.vertex.vertices, self.instance.vertices)}
                    for v in self.vertex.vertices:
                        v.exact_match_fragment = match_fragment_dict[v]
                        for fpath in v.exact_match_fragment:
                            (f, path) = fpath
                            new_fpath = (f, tuple(vertex_map[v0] for v0 in path))
                            vertex_map[v].exact_match_fragment.add(new_fpath)
            return self.instance

        def save_instance(self):
            """Save an instance of resource vertex to cache
            """
            # save current instance to cache
            self.vertex_cache.append(self.instance)
            # restore an instance in use
            self.instance = self.vertex_in_use.pop()

        def clear_cache(self):
            """Clear cache of instance and fragment count
            """
            self.instance = None
            self.fragment_count = Counter()
            self.fragment_path = list()
            self.vertex_cache = list()
            self.vertex_in_use = list()

    def __init__(self):
        """Constructor of GraphGenResourceManager
        """
        self.symbol_res_map = dict()
        self.def_symbol_map = dict()

    def has_resource(self, resource):
        """Check if a resource is registered to the manager

        Args:
            resource (GraphGenResourceManager.Resource): a resource for the structure generation

        Returns:
            bool: True is registered
        """
        return resource.get_symbol() in self.symbol_res_map

    def add_resource(self, definition, resource):
        """Register new vertex with its definition

        Args:
            definition (str, tuple): definition of atom/ring resource
            resource (GraphGenResourceManager.Resource): a resource for the structure generation

        Returns:
            bool: True if successfully added
        """
        if resource.get_symbol() in self.symbol_res_map:
            logger.warning('duplicated resource symbol:{0} def={1}'.format(resource.get_symbol(), definition))
            return False
        self.symbol_res_map[resource.get_symbol()] = resource
        self.def_symbol_map[definition] = resource.get_symbol()
        return True

    def add_dup_resource(self, definition, resource):
        """Register duplicated vertex with its definition

        Args:
            definition(str, tuple): definition of atom/ring resource
            resource (GraphGenResourceManager.Resource): a resource for the structure generation
        """
        # since there is duplicate definition for the same symbol
        # register only to def_symbol_map
        self.def_symbol_map[definition] = resource.get_symbol()

    def get_resource_from_definition(self, definition):
        """Get a resource from a definition

        Args:
            definition (str, tuple): definition of a resource

        Returns:
            GraphGenResourceManager.Resource: a resource object
        """
        resource = self.symbol_res_map.get(self.get_symbol(definition), None)
        return resource

    def get_resource(self, symbol):
        """Get a resource from a symbol

        Args:
            symbol (str): symbol of a resource

        Returns:
            GraphGenResourceManager.Resource: a resource object
        """
        resource = self.symbol_res_map.get(symbol, None)
        if resource is None:
            logger.warning('no vertex resource registered for symbol:{0}'.format(symbol))
        return resource

    def get_symbol(self, definition):
        """Get a symbol of a resource from a definition

        Args:
            definition (str, tuple): definition of a resource

        Returns:
            str: a symbol of a resource
        """
        symbol = self.def_symbol_map.get(definition, None)
        if symbol is None:
            logger.warning('no vertex resource registered for definition:{0}'.format(definition))
        return symbol

    def get_symbols(self):
        """Get a list of symbols registered to the resource manager

        Returns:
            list: a list of symbols of vertex resources
        """
        return list(self.symbol_res_map.keys())

    def clear_cache(self):
        """Clear cache of all the resources
        """
        for resource in self.symbol_res_map.values():
            resource.clear_cache()


class AtomResManager(GraphGenResourceManager):
    """Factory and manager of atom resources in the structure generation.

    Attributes:
        atom_valence_map (dict): a dictionary of a valence of an atom
    """
    class Resource(GraphGenResourceManager.Resource):
        """An atom vertex as a resource of the structure generation

        Attributes:
            valence (int): valence of an atom resource
        """
        def __init__(self, atom_definition):
            """Constructor of AtomResManager.Resource from atom definition.
            Atom definition should be
            * atom_symbol :str
            * (atom_symbol, valence) :tuple

            Args:
                atom_definition (str, tuple): definition of atom resource
            """
            self.valence = 0

            if isinstance(atom_definition, tuple):
                (atom_symbol, valence) = atom_definition
            else:
                atom_symbol = atom_definition
                valence = None
            atom_vertex = AtomVertex(0, atom_symbol)
            super().__init__(atom_symbol, atom_vertex)
            self.atom_count[atom_symbol] = 1
            self.valence = valence

        def get_valence(self):
            """Get valence of an atom

            Returns:
                int: valence
            """
            return self.valence

    def __init__(self):
        """Constructor of AtomResManager.
        """
        super().__init__()
        self.atom_valence_map = dict()

    def create_resources(self, atom_definitions):
        """Create atom resources from definitions

        Args:
            atom_definitions (list): a list of atom definitions
        """
        for atom_def in atom_definitions:
            # create an atom vertex object of atom resource
            atom_resource = self.Resource(atom_def)
            # update valence map from a new atom vertex object
            atom_symbol = atom_resource.get_symbol()
            atom_valence = atom_resource.get_valence()
            if atom_valence is not None:
                if atom_valence <= 0:
                    logger.error('not positive valence atom is ignored: {0}'.format(atom_valence))
                    continue
                if atom_symbol not in self.atom_valence_map:
                    self.atom_valence_map[atom_symbol] = atom_valence
                else:
                    logger.warning('duplicated valence for atom: {0}'.format(atom_symbol))
                    self.atom_valence_map[atom_symbol] = max(self.atom_valence_map[atom_symbol], atom_valence)
                atom_resource.get_vertex().valence = atom_valence
            # register new atom resource object
            if logger.isEnabledFor(logging.INFO):
                logger.info('register atom resource:{0} def={1}'.format(atom_symbol, atom_def))
            self.add_resource(atom_def, atom_resource)

    def create_resource(self, atom_symbol, valence=None):
        """Create an atom resource by symbol and valence, and returns a resource object

        Args:
            atom_symbol (str): symbol of atom resource
            valence(int, optional): valence of atom. Defaults to None.

        Returns:
            AtomResManager.Resource: atom resource
        """
        if atom_symbol in self.symbol_res_map:
            resource = self.symbol_res_map[atom_symbol]
            if valence is not None:
                if valence > resource.valence:
                    resource.valence = valence
                    self.atom_valence_map[atom_symbol] = valence
            return resource
        else:
            if valence is not None:
                atom_def = (atom_symbol, valence)
                self.atom_valence_map[atom_symbol] = valence
            else:
                atom_def = atom_symbol
            resource = self.Resource(atom_def)
            self.add_resource(atom_def, resource)
            return resource

    def update_valence(self, atom_graph):
        """Update valence of vertices in a AtomGraph object

        Args:
            atom_graph (AtomGraph): an AtomGraph object
        """
        for v in atom_graph.vertices:
            if v.atom in self.atom_valence_map and not v.aromatic_atom():
                v.valence = self.atom_valence_map[v.atom]

    def get_valence(self, atom_symbol):
        """Get valence of atom symbol

        Args:
            atom_symbol (str): atom symbol

        Returns:
            int: valence of atom
        """
        if atom_symbol in self.symbol_res_map:
            return self.symbol_res_map[atom_symbol].get_vertex().num_valence()
        else:
            logger.error('no nnvalence of atom:{0}'.format(atom_symbol))
            return 0


class RingResManager(GraphGenResourceManager):
    """Factory and manager of ring resources in the structure generation.

    Attributes:

    """

    class Resource(GraphGenResourceManager.Resource):
        """A ring resource of the structure generation

        Attributes:
            graph (AtomGraph): graph of a ring
            labeling (ChemGraphLabeling): a labeling of ring graph
            replacement (dict): a dictionary of replacing atoms and ranges
            extension (dict): a dictionary of extension points and automorphism of a connecting point
            members (list): a list of ring resources generated by replacement
            group_base (RingResManager.Resource): base ring of a ring generated by replacement
            group_atoms (set): atoms in a base ring
            group_atom_count (Counter): a counter of atoms in a ring group
        """

        def __init__(self, ring_definition, atom_res_mgr):
            """Constructor of RingResManager.Resource from ring definition.
            Ring definition should be
            * ring_smiles :str
            * (ring_smiles, replacement) :tuple

            Args:
                ring_definition (str, tuple): definition of ring resource
                atom_res_mgr (AtomResManager): atom resource manager
            """
            self.graph = None
            self.labeling = None
            self.replacement = dict()
            self.extension = dict()
            self.members = list()
            self.group_base = None
            self.group_atoms = set()
            self.group_atom_count = Counter()

            if isinstance(ring_definition, tuple):
                (ring_smiles, t_replacement) = ring_definition
                replacement = dict(t_replacement)
            else:
                ring_smiles = ring_definition
                replacement = None
            # get canonical smiles as ring symbol
            ring_symbol = ChemGraph.canonical_smiles(ring_smiles)
            if ring_symbol is None:
                raise (ValueError('invalid smiles for a ring resource:{0}'.format(ring_smiles)))
            ring_graph = AtomGraph(smiles=ring_symbol)
            # reorder vertices by canonical smiles order
            ring_graph.reorder_canonical()
            # update valence
            atom_res_mgr.update_valence(ring_graph)
            ring_vertex = RingAsVertex(0, ring_symbol, ring_graph, 0)
            # initialize resource object
            super().__init__(ring_symbol, ring_vertex)
            self.atom_count = ring_vertex.atom_count
            self.ring_count = ring_vertex.ring_count
            self.aromatic_count = ring_vertex.aromatic_count
            self.graph = ring_graph
            self.replacement = replacement

            if replacement is not None:
                # a base ring, create member resources by replacing atoms of base ring
                member_graphs = ChemRingGenerator(ring_graph).generate_ring_graphs(replacement)
                member_symbols = set()
                for member_graph in member_graphs:
                    member_symbol = ChemGraph.canonical_smiles(member_graph.to_smiles())
                    if member_symbol not in member_symbols:
                        member_resource = self.__class__(member_symbol, atom_res_mgr)
                        member_resource.group_base = self
                        self.members.append(member_resource)
                        member_symbols.add(member_resource.get_symbol())
                    else:
                        logger.warning('duplicated member ring generation: {0} from {1}'.
                                       format(member_symbol, ring_definition))
                # get group atom count
                for member in self.members:
                    ring_vertex = member.get_vertex()
                    for atom, count in ring_vertex.atom_count.items():
                        self.group_atoms.add(atom)
                        if atom in self.group_atom_count:
                            self.group_atom_count[atom] = min(self.group_atom_count[atom], count)
                        else:
                            self.group_atom_count[atom] = count
            else:
                # not a base ring, prepare as a ring resource for the structure generation
                self.labeling = ChemGraphLabeling(ring_vertex.vertices)
                # find extension points
                for v in ring_vertex.vertices:
                    if v.num_all_free_hand() < 0:
                        continue
                    if self.labeling.automorphism.is_min_orbit(v.index):
                        # find positions to extend if v is selected as connection point
                        old_symbol = v.color()
                        v.set_color('')
                        lv = ChemGraphLabeling(ring_vertex.vertices)
                        self.extension[v.index] = ([v0.index for v0 in ring_vertex.vertices
                                                   if lv.automorphism.is_min_orbit(v0.index)], lv.automorphism)
                        v.set_color(old_symbol)

        def get_graph(self):
            """Get original graph of a ring

            Returns:
                AtomGraph: a graph of ring
            """
            return self.graph

        def is_base_ring(self):
            """Check if it is a base ring resource

            Returns:
                bool: True if a base ring
            """
            return self.replacement is not None

        def get_replacement(self):
            """Get atom replacement for base ring

            Returns:
                tuple: atom replacement definition
            """
            return self.replacement

        def get_labeling(self):
            """Get labeling of ring resource

            Returns:
                ChemGraphLabeling: labeling
            """
            return self.labeling

        def get_extension(self):
            """Get extension points of ring resource

            Returns:
                tuple: extension points (vertex index, automorphism)
            """
            return self.extension

        def get_group_atom_count(self):
            """Get atom counts of members of a base ring resource

            Returns:
                Counter: counter of atoms
            """
            return self.group_atom_count

        def get_members(self):
            """Get member ring resource of base ring resource

            Returns:
                list: a plist of ring resources
            """
            return self.members

        def get_base_resource(self):
            """Get base ring resource of a ring resource

            Returns:
                RingResManager.Resource: a base ring resource
            """
            return self.group_base

        def get_base_symbol(self):
            """Get symbol of a base ring resource of a ring resource

            Returns:
                str: symbol of a base ring resource
            """
            if self.group_base is not None:
                return self.group_base.get_symbol()
            else:
                return None

    def __init__(self):
        """Constructor of RingResManager.
        """
        super().__init__()
        self.base_symbol_res_map = dict()
        self.base_def_symbol_map = dict()
        self.base_symbol_map = dict()

    def create_resources(self, ring_definitions, atom_res_mgr):
        """Create ring resources from definitions

        Args:
            ring_definitions (list): a list of ring definitions
            atom_res_mgr (AtomResManager): atom resource manager
        """

        for ring_def in ring_definitions:
            # create a ring vertex object of ring resource
            try:
                ring_resource = self.Resource(ring_def, atom_res_mgr)
            except ValueError as e:
                logger.error('ring resource: {0}'.format(e))
                continue

            if not ring_resource.is_base_ring():
                # register ring resource object
                if ring_resource.get_symbol() not in self.symbol_res_map:
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('register ring resource:{0} def={1}'.
                                    format(ring_resource.get_symbol(), ring_def))
                    self.add_resource(ring_def, ring_resource)
                else:
                    base_res_symbol = self.base_symbol_map.get(ring_resource.get_symbol(), None)
                    if logger.isEnabledFor(logging.INFO):
                        if base_res_symbol is None:
                            logger.info('ring {0} is already created'.format(ring_resource.get_symbol()))
                        else:
                            logger.info('ring {0} is already created as member of base ring {1}'.
                                        format(ring_resource.get_symbol(), base_res_symbol))

            else:
                # register base ring resource
                if ring_resource.get_symbol() not in self.base_symbol_res_map:
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('register base ring resource:{0} def={1}'.
                                    format(ring_resource.get_symbol(), ring_def))
                    self.add_base_resource(ring_def, ring_resource)
                else:
                    dup_resource = self.get_base_resource(ring_resource.get_symbol())
                    # check the consistency of connection condition
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('base ring {0} is already created'.
                                    format(ring_resource.get_symbol()))
                    if dup_resource.replacement == ring_resource.replacement:
                        self.add_dup_base_resource(ring_def, ring_resource)
                    else:
                        logger.warning('inconsistent duplicated ring resource:{0} def={1}'.
                                       format(ring_resource.get_symbol(), ring_def))
                        continue

                # register member ring resources created by replacement
                for member_res in ring_resource.get_members():
                    if member_res.get_symbol() not in self.symbol_res_map:
                        if logger.isEnabledFor(logging.INFO):
                            logger.info('register member ring resource:{0} base={1}'.
                                        format(member_res.get_symbol(), ring_resource.get_symbol()))
                        self.add_resource(member_res.get_symbol(), member_res)
                        self.base_symbol_map[member_res.get_symbol()] = ring_resource.get_symbol()
                    else:
                        base_res_symbol = self.base_symbol_map.get(member_res.get_symbol(), None)
                        if logger.isEnabledFor(logging.INFO):
                            if base_res_symbol is None:
                                logger.info('member {0} of base ring {1} is already created'.
                                            format(member_res.get_symbol(), ring_resource.get_symbol()))
                            else:
                                logger.info('member {0} of base ring {1} is already created for base ring {2}'.
                                            format(member_res.get_symbol(), ring_resource.get_symbol(),
                                                   base_res_symbol))

    def create_resource(self, ring_symbol, atom_res_mgr):
        """Create a ring resource from ring symbol, and return a ring resource

        Args:
            ring_symbol (str): ring symbol
            atom_res_mgr (AtomResManager): atom resource manager

        Returns:
            RingResManager.Resource: a ring resource
        """
        if ring_symbol in self.symbol_res_map:
            resource = self.symbol_res_map[ring_symbol]
            return resource
        else:
            resource = self.Resource(ring_symbol, atom_res_mgr)
            self.add_resource(resource.get_symbol(), resource)
            return resource

    def add_base_resource(self, definition, resource):
        """Register new base ring resource with its definition

        Args:
            definition (str, tuple): definition of base ring resource
            resource (RingResManager.Resource): a resource for the structure generation

        Returns:
            bool: True if successfully added
        """
        if resource.get_symbol() in self.base_symbol_res_map:
            logger.warning('duplicated base ring resource symbol:{0} def={1}'.format(resource.get_symbol(), definition))
            return False
        self.base_symbol_res_map[resource.get_symbol()] = resource
        self.base_def_symbol_map[definition] = resource.get_symbol()
        return True

    def add_dup_base_resource(self, definition, resource):
        """Register duplicated vertex with its definition

        Args:
            definition(str, tuple): definition of atom/ring resource
            resource (GraphGenResourceManager.Resource): a resource for the structure generation
        """
        # since there is duplicate definition for the same symbol
        # register only to def_symbol_map
        self.base_def_symbol_map[definition] = resource.get_symbol()

    def get_resource_from_definition(self, definition):
        """Get a (base) ring resource from a definition

        Returns:
            GraphGenResourceManager.Resource: a resource object
        """
        if definition in self.base_def_symbol_map:
            return self.get_base_resource_from_definition(definition)
        elif definition in self.def_symbol_map:
            return super().get_resource_from_definition(definition)
        else:
            return None

    def get_base_resource_from_definition(self, definition):
        """Get a base ring resource from a definition

        Args:
            definition (str, tuple): definition of a resource

        Returns:
            GraphGenResourceManager.Resource: a resource object
        """
        resource = self.base_symbol_res_map.get(self.get_base_symbol(definition), None)
        return resource

    def get_base_resource(self, symbol):
        """Get a base ring resource from a symbol

        Args:
            symbol (str): symbol of a resource

        Returns:
            GraphGenResourceManager.Resource: a resource object
        """
        resource = self.base_symbol_res_map.get(symbol, None)
        if resource is None:
            logger.warning('no base ring resource registered for symbol:{0}'.format(symbol))
        return resource

    def get_base_symbol(self, definition):
        """Get a symbol of a base ring resource from a definition

        Args:
            definition (str, tuple): definition of a resource

        Returns:
            str: a symbol of a resource
        """
        symbol = self.base_def_symbol_map.get(definition, None)
        if symbol is None:
            logger.warning('no base ring resource registered for definition:{0}'.format(definition))
        return symbol

    def get_base_symbols(self):
        """Get a list of symbols registered to the resource manager

        Returns:
            list: a list of symbols of vertex resources
        """
        return list(self.base_symbol_res_map.keys())
