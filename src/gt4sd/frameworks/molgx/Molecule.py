# -*- coding:utf-8 -*-
"""
Molecule.py

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

from rdkit import Chem
from rdkit.Chem import Descriptors

from .ChemGenerator.ChemGraph import AtomGraph
from .ChemGenerator.ChemGraphLabeling import ChemGraphLabeling

import numpy as np
import pandas as pd

import copy
from enum import Enum
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# Molecule: a class of molecule
# -----------------------------------------------------------------------------

class MolType(Enum):
    """Enum class for molecule types
    """
    SIMPLE = 0


class Molecule (object):
    """Base class of molecule objects. Properties and features of the molecule is stored to this object
    with a dictionary.

    Attributes:
        id (str): unique id of a molecule
        property_map (dict): a map from property to its value
        feature_map (dict): a map from feature to its value
    """
    def __init__(self, id):
        """Constructor of Molecule

        Args:
            id (str): unique id of a molecule
        """
        self.id = id
        self.property_map = dict()
        self.feature_map = dict()

    def get_id(self):
        """Get id of a molecule.

        Returns:
            str: id of a molecule
        """
        return self.id

    def has_property(self, prop):
        """Check if a molecule has a given property.

        Args:
            prop (Property): property object

        Returns:
            bool: true if a molecule has a property
        """
        return prop.id in self.property_map

    def set_property(self, prop, value):
        """Set a property and value to a molecule.

        Args:
            prop (Property): property object
            value (object): property value
        """
        self.property_map[prop.id] = value

    def get_property(self, prop):
        """Get a property value of a molecule.

        Args:
            prop (Property): property object

        Returns:
            object: property value
        """
        return self.property_map[prop.id]

    def remove_property(self, prop):
        """Remove a property value of a molecule.

        Args:
            prop (Property): property object
        """
        if prop.id in self.property_map:
            del self.property_map[prop.id]

    def has_feature(self, feature):
        """Check if a molecule has a given feature.

        Args:
            feature (FeatureExtraction.Feature): feature object

        Returns:
            bool: true if a molecule has a feature
        """
        return feature.id in self.feature_map

    def set_feature(self, feature, value):
        """Set a feature and value to a molecule.

        Args:
            feature (FeatureExtraction.Feature): feature object
            value (object): feature value
        """
        self.feature_map[feature.id] = value

    def get_feature(self, feature):
        """Get a feature value of a molecule.

        Args:
            feature (Features): feature object

        Returns:
            object: feature value
        """
        return self.feature_map[feature.id]

    def remove_feature(self, feature):
        """Remove a feature from a molecule

        Args:
            feature (Feature): feature object
        """
        if feature.id in self.feature_map:
            del self.feature_map[feature.id]

    def clear_feature(self):
        """Remove all the features from a molecule
        """
        self.feature_map.clear()

    def info(self, features_list=None):
        """Print information of a molecule, which includes properties and features.

        Args:
            features_list (list, optional): a list of features. Defaults to None
        """
        print('Molecule: id={0}'.
              format(self.get_id()))
        print('Properties:')
        print('  {0}'.format(self.property_map))
        if features_list is not None:
            for features in features_list:
                feature_map = dict()
                for f in features.get_feature_list():
                    if self.has_feature(f):
                        feature_map[f.id] = self.get_feature(f)
                if len(feature_map) > 0:
                    print('FeatureSet:{0}'.format(features.id))
                    print('  {0}'.format(feature_map))


class SimpleMolecule(Molecule):
    """Wrapper class of rdkit mol object.

    Attributes:
        mol (Mol): rdkit Mol object
        graph (AtomGraph): graph representation of the molecule
        original_mol: original mol object
        original_smiles: original smiles string
        anion (Mol): anion part of molecule
        cation (Mol): cation part molecule
        main_ion (Mol): main ion part
        sub_ion (Mol): sub ion part
        mol_block (str): text of mol block from SDF
    """
    def __init__(self, id, mol=None, graph=None, smiles=None, smarts=None, mol_block=None):
        """Constructor of SimpleMolecule instance.

        Args:
            id (str): unique id of a molecule
            mol (Mol, optional): rdkit Mol object. Defaults to None.
            graph (AtomGraph): graph. Defaults to None.
            smiles (str, optional): SMILES of molecule. Defaults to None.
            smarts (str, optional): SMARTS of molecule. Defaults to None.

        Note:
            One of mol, smiles, or smart must be specified
        """
        super().__init__(id)
        self.graph = None
        self.graph_labeling = None
        self.original_smiles = None
        if mol is not None:
            self.mol = mol
        elif graph is not None:
            self.graph = graph
            self.mol = graph.to_mol()
        elif smiles is not None:
            self.original_smiles = smiles
            self.mol = Chem.MolFromSmiles(smiles)
            if self.mol is None:
                logger.error('SimpleMolecule: RDKit failed to read SMILES:[%s] %s', id, smiles)
        elif smarts is not None:
            self.mol = Chem.MolFromSmarts(smarts)
            if self.mol is None:
                logger.error('SimpleMolecule: RDKit failed to read SMARTS:[%s] %s', id, smarts)
        else:
            self.mol = None
            logger.error('SimpleMolecule: mol, smiles or smarts are necessary for object creation')
        self.original_mol = self.mol
        self.anion = None
        self.cation = None
        self.main_ion = None
        self.sub_ion = None
        self.mol_block = mol_block

    def get_mol(self):
        """Get rdkit Mol object.

        Returns:
             Mol: rdkit Mol object
        """
        return self.mol

    def get_smiles(self):
        """Get SMILES representation of a molecule.

        Returns:
            str: SMILES representation
        """
        return Chem.MolToSmiles(self.mol)

    def get_original_smiles(self):
        """Get original SMILES representation of a molecule without sanitization.

        Returns:
            str: SMILES representation
        """
        return Chem.MolToSmiles(self.mol)

    def get_smarts(self):
        """Get SMARTS representation of a molecule.

        Returns:
            str: SMARTS representation
        """
        return Chem.MolToSmarts(self.mol)

    def get_graph(self):
        """Get graph representation of a molecule.

        Returns:
            AtomGraph: graph representation
        """
        if self.graph is None:
            self.graph = AtomGraph(mol=self.mol)
        return self.graph

    def get_graph_labeling(self):
        """Get canonical labeling of graph representation

        Returns:
            ChemGraphLabeling: graph labeling
        """
        if self.graph_labeling is None:
            graph = self.get_graph()
            self.graph_labeling = ChemGraphLabeling(graph.vertices)
        return self.graph_labeling

    def get_original_mol(self):
        """Get original rdkit Mol

        Returns:
            Mol: rdkit Mol object
        """
        return self.original_mol

    def get_anion(self):
        """Get anion part of molecule

        Returns:
            Mol: rdkit Mol object
        """
        return self.anion

    def get_cation(self):
        """Get cation part of molecule

        Returns:
            Mol: rdkit Mol object
        """
        return self.cation

    def get_main_ion(self):
        """Get main ion part of molecule

        Returns:
            Mol: rdkit Mol object
        """
        return self.main_ion

    def get_sub_ion(self):
        """Get sub ion part of molecule

        Returns:
            Mol: rdkit Mol object
        """
        return self.sub_ion

    def get_mol_block(self):
        """Get mol block text from SDF

        Returns:
            str: mol block
        """
        return self.mol_block

    def get_weight(self):
        """Get mol weight of a molecule

        Returns:
            float: mol weight
        """
        return Descriptors.ExactMolWt(self.mol)

    def ion_separation(self):
        """Separate a molecule into anion and cation

        Returns:
            bool: flag of successful separation
        """
        ions = defaultdict(list)
        used_atom_index = set()
        found_smiles = set()
        for atom in self.mol.GetAtoms():
            if atom.GetIdx() not in used_atom_index:
                # collect connected component
                charge = 0
                connected_atom_indices = set()
                used_atom_index.add(atom.GetIdx())
                new_atoms = [atom]
                while len(new_atoms) > 0:
                    a = new_atoms.pop(-1)
                    charge += a.GetFormalCharge()
                    connected_atom_indices.add(a.GetIdx())
                    for b in a.GetBonds():
                        new_a = b.GetEndAtom() if b.GetBeginAtomIdx() == a.GetIdx() else b.GetBeginAtom()
                        if new_a.GetIdx() not in used_atom_index:
                            new_atoms.append(new_a)
                            used_atom_index.add(new_a.GetIdx())
                if len(connected_atom_indices) < self.mol.GetNumAtoms():
                    # make ion molecule
                    ion = Chem.RWMol(self.mol)
                    for index in reversed(range(self.mol.GetNumAtoms())):
                        if index not in connected_atom_indices:
                            ion.RemoveAtom(index)
                    mol = ion.GetMol()
                    molecule = SimpleMolecule(self.get_id(), mol=mol)
                    if molecule.get_smiles() not in found_smiles:
                        ions[charge].append(molecule)
                        found_smiles.add(molecule.get_smiles())
        if len(ions) > 0:
            anions = []
            cations = []
            neutrals = []
            for charge in ions:
                if charge < 0:
                    anions.extend(ions[charge])
                elif charge > 0:
                    cations.extend(ions[charge])
                else:
                    neutrals.extend(ions[charge])
            if len(anions) > 0:
                self.anion = anions[0]
                if len(cations) > 0:
                    self.cation = cations[0]
                else:
                    if len(neutrals) > 0:
                        self.cation = neutrals[0]
                        logger.warning('no cation, neutral molecule assigned: %s', self.get_id())
                    else:
                        self.cation = SimpleMolecule(self.get_id(), smiles='')
                        logger.warning('no cation, empty molecule assigned: %s', self.get_id())
            else:
                if len(cations) > 0:
                    self.cation = cations[0]
                    if len(neutrals) > 0:
                        self.anion = neutrals[0]
                        logger.warning('no anion, neutral molecule assigned: %s', self.get_id())
                    else:
                        self.anion = SimpleMolecule(self.get_id(), smiles='')
                        logger.warning('no cation, empty molecule assigned: %s', self.get_id())
                else:
                    if len(neutrals) > 1:
                        self.cation = neutrals[0]
                        self.anion = neutrals[1]
                        logger.warning('no cation/anion, neutral molecules assigned to both: %s', self.get_id())
                    elif len(neutrals) > 0:
                        self.cation = neutrals[0]
                        self.anion = SimpleMolecule(self.get_id(), smiles='')
                        logger.warning('no cation/anion, neutral and empty molecule assigned to cation and cation: %s',
                                       self.get_id())
                    else:
                        self.cation = SimpleMolecule(self.get_id(), smiles='')
                        self.anion = SimpleMolecule(self.get_id(), smiles='')
                        logger.warning('no cation/anion, empty molecules assigned to both: %s',
                                       self.get_id())
            return True
        else:
            self.anion = SimpleMolecule(self.get_id(), smiles='')
            self.cation = SimpleMolecule(self.get_id(), smiles='')
            return False

    def ion_separation_by_matching(self, fragment):
        """Separate a molecule into main/sub ion by the fragment matching

        Returns:
            bool: flag of successful separation
        """
        if self.anion is None and self.cation is None:
            if not self.ion_separation():
                self.main_ion = SimpleMolecule(self.get_id(), mol=self.mol)
                self.sub_ion = SimpleMolecule(self.get_id(), smiles='')
                return False
        # compare the matching count
        anion_count = fragment.count_fragment_graph(AtomGraph(mol=self.anion.get_mol()))
        cation_count = fragment.count_fragment_graph(AtomGraph(mol=self.cation.get_mol()))
        if anion_count > cation_count:
            self.main_ion = self.anion
            self.sub_ion = self.cation
        elif anion_count < cation_count:
            self.main_ion = self.cation
            self.sub_ion = self.anion
        else:
            if self.anion.get_weight() > self.cation.get_weight():
                self.main_ion = self.anion
                self.sub_ion = self.cation
            else:
                self.main_ion = self.cation
                self.sub_ion = self.anion
        return True

    def ion_separation_by_weight(self):
        """Separate a molecule into main/sub ion by weight

        Returns:
            bool: flag of successful separation
        """
        if self.anion is None and self.cation is None:
            if not self.ion_separation():
                self.main_ion = SimpleMolecule(self.get_id(), smiles=self.mol)
                self.sub_ion = SimpleMolecule(self.get_id(), smiles='')
                return False
        # compare the molecule weight
        if self.anion.get_weight() >= self.cation.get_weight():
            self.main_ion = self.anion
            self.sub_ion = self.cation
        else:
            self.main_ion = self.cation
            self.sub_ion = self.anion
        return True

    def get_ring_atom_count(self):
        """Get the number of atoms in a ring

        Returns:
            int: the number of atoms
        """
        return self.get_graph().get_ring_atom_count()

    def print(self, features_list=None):
        """Print information of a molecule, which includes properties and features.

        Args:
            features_list (list, optional): a list of features. Defaults to None
        """
        print('Molecule: id={0} SMILES={1} SMARTS={2}'.
              format(self.get_id(), self.get_smiles(), self.get_smarts()))
        print('Properties:')
        print('  {0}'.format(self.property_map))
        if features_list is not None:
            for features in features_list:
                feature_map = dict()
                for f in features.get_feature_list():
                    if self.has_feature(f):
                        feature_map[f.id] = self.get_feature(f)
                if len(feature_map) > 0:
                    print('FeatureSet:{0}'.format(features.id))
                    print('  {0}'.format(feature_map))


class GeneratedMolecule(SimpleMolecule):
    """Class of molecule generated by structure generator

    Attributes:
        expand_graph (AtomGraph: graph without wild card atom
        generation_path (str): generation path of atoms
        vector_candidate (FeatureEstimationResult.Candidate): feature vector for generation
    """

    def __init__(self, index, graph, generation_path, vector_candidate,
                 mol_type):
        """Constructor of GeneratedMolecule

            index (int): index of a molecule in vector_candidate
            graph (AtomGraph): generated molecule graph
            generation_path (str): generation path of atoms
            vector_candidate (FeatureEstimationResult.Candidate): feature vector for generation
            mol_type (MolType): type of a molecule
        """
        # keep original mol
        self.expand_graph = copy.deepcopy(graph)
        if vector_candidate is None:
            super().__init__('M{0}'.format(index), mol=graph.translate_to_mol())
        else:
            super().__init__('{0}M{1}'.format(vector_candidate.get_id(), index), mol=graph.translate_to_mol())
        self.graph = graph
        self.generation_path = generation_path
        self.vector_candidate = vector_candidate
        self.mol_type = mol_type

    def get_label(self):
        """Get a label of component

        Returns:
            str: label
        """
        return self.vector_candidate.get_label()

    def get_mol_type(self):
        """Get a mol type of generation context

        Returns:
            MolType: a mol type
        """
        if self.vector_candidate is not None:
            return self.vector_candidate.get_mol_type()
        else:
            return None

    def get_expand_mol(self):
        """Get RDKit object with wild card replaced by atom

        Returns:
            Mol: rdkit mol
        """
        return self.expand_graph.translate_to_mol()

    def get_generation_path(self):
        """Get a string sequence of atoms for generating molecule

        Returns:
            str: str of atom symbols
        """
        return self.generation_path

    def get_generation_path_list(self):
        """Get a sequence of atoms for generating molecule

        Returns:
            list: list of atom symbols
        """
        return self.generation_path.lstrip('/').split('/')

    def get_vector_candidate(self):
        """Get a candidate of feature vector

        Returns:
            FeatureEstimationResult.Candidate: feature vector
        """
        return self.vector_candidate


# -----------------------------------------------------------------------------
# Property: property of a molecule
# -----------------------------------------------------------------------------

class Property(object):
    """Class of property of a molecule.

    Attributes:
        id (str): unique name of a property
    """

    def __init__(self, id, dtype=object):
        """Constructor of a property.

        Args:
            id (str): unique name of a property
        """
        self.id = id
        self.dtype = dtype

    def get_id(self):
        """Get an id of a property.

        Returns:
            id (str): id of a property
        """
        return self.id

    def get_dtype(self):
        """Get data type of a property

        Returns:
            type: data type
        """
        return self.dtype


# -----------------------------------------------------------------------------
# PropertySet: a set of properties of a molecule
# -----------------------------------------------------------------------------

class PropertySet(object):
    """Class of a set of properties defined for a molecule.

    Attributes:
        properties (list): a list of Property objects
        id_map (dict): a map of property id to an index in a list of Property objects
    """

    def __init__(self, properties=None):
        """Constructor of a property set.

        Args:
            properties (list): list of Property objects
        """
        self.properties = []
        self.id_map = dict()
        if properties is not None:
            for p in properties:
                self.add_property(p)

    def has_property(self, id):
        """Check if a property is included in a property set.

        Args:
            id (str): id of a property

        Returns:
            bool: true is a property is included in a property set
        """
        return id in self.id_map

    def add_property(self, prop):
        """Add a property to a property set.

        Args:
            prop (Property): a property object
        """
        self.properties.append(prop)
        self.id_map[prop.id] = prop

    def get_property_list(self):
        """Get a list of Property objects.

        Returns:
            list: a list of Property objects
        """
        return self.properties

    def sort_property(self):
        """Sort property by its id.
        """
        self.properties.sort(key=lambda x: x.id)

    def get_header_list(self):
        """Get a list of property ids as a header of csv file.

        Returns:
            list: a list of property ids
        """
        return [m.id for m in self.properties]

    def make_property_vector(self, mols):
        """Get a property vector from a set of properties.

        Args:
            mols (list): a list of molecules

        Return:
            DataFrame: array of property vectors of a molecule (mols x properties)
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('property vector: mols=%d property=%d', len(mols), len(self.properties))
        # make a feature vector by counting features in a molecule
        property_vector = np.zeros((len(mols), len(self.properties)), dtype=object)
        for (i, mol) in enumerate(mols):
            for (j, prop) in enumerate(self.properties):
                if mol.has_property(prop):
                    property_vector[i][j] = mol.get_property(prop)
                else:
                    property_vector[i][j] = 0
        df_index = [m.id for m in mols]
        dataframe = pd.DataFrame(data=property_vector, index=df_index, columns=self.get_header_list())
        return dataframe
