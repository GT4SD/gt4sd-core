# -*- coding:utf-8 -*-
"""
FeatureExtraction.py

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

import inspect
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem

from .ChemGenerator.ChemGraph import AtomGraph, ChemVertex
from .ChemGenerator.ChemGraphFragment import ChemFragment
from .ChemGenerator.ChemGraphGen import AtomGraphNode
from .ChemGenerator.ChemGraphLabeling import ChemGraphLabeling
from .Molecule import *
from .Utility import get_subclasses

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def print_feature_extractor():
    """Print all the available feature extractors.
    """
    classes = get_subclasses(FeatureExtractor)
    print('Available feature extractors:')
    for index, cls in enumerate(classes):
        print('{0}: {1}'.format(index, cls.__name__))


def get_feature_extractor():
    """Get all the available feature extractors.

    Returns:
        dict: a mapping of class name and class object
    """
    class_map = dict()
    classes = get_subclasses(FeatureExtractor)
    for cls in classes:
        class_map[cls.__name__] = cls
    return class_map


# -----------------------------------------------------------------------------
# Feature: a feature of molecule
# -----------------------------------------------------------------------------

class Feature(object):
    """Base class of molecule features.

    Attributes:
        id (str): string id
        shape(tuple): shape of feature vector
    """

    dtype = None
    default_val = None

    def __init__(self, feature_id, value_id, shape=()):
        """Constructor of Feature class.

        Args:
            feature_id (str): name of feature
            value_id (object): feature value object
        """
        self.id = self.get_id_string(feature_id, value_id)
        self.shape = shape

    def get_id(self):
        """Get id of the feature.

        Returns:
            str: id of the feature
        """
        return self.id

    def get_index(self):
        """Get index of dictionary.

        Returns:
            str, int: index
        """
        return self.id

    def get_shape(self):
        """Get shape of the feature vector

        Returns:
            tuple: shape
        """
        return self.shape

    def get_vector_size(self):
        """Get the size of feature vector.

        Returns:
            int: size of feature vector
        """
        vector_size = 1
        for size in self.shape:
            vector_size *= size
        return vector_size

    def get_header_list(self):
        """Get header of feature vector in a csv file.

        Returns:
            list: a list of header names
        """
        if len(self.shape) == 0:
            return [self.id]
        else:
            header_list = ['{0}['.format(self.id)]
            for dim in range(len(self.shape)):
                new_header_list = []
                for header in header_list:
                    for index in range(self.shape[dim]):
                        if dim + 1 == len(self.shape):
                            new_header_list.append('{0}{1:d}]'.format(header, index))
                        else:
                            new_header_list.append('{0}{1:d},'.format(header, index))
                header_list = new_header_list
        return header_list

    def get_dtype(self):
        """Get data type of feature vector

        Returns:
            dtype: data type
        """
        return self.dtype

    def get_shaped_value(self, value):
        """Get values in shaped arrays

        Returns:
            list: a list of tuples (shape, value, default_value)
        """
        return [(self.shape, value, self.default_val)]

    def get_formatted_value(self, value):
        """Get value in readable string

        Returns:
            str: a formatted string value
        """
        return format(value)

    def get_default_value(self):
        """Get default value of feature vector

        Returns:
            object: default value
        """
        return self.default_val

    def get_domain(self):
        """Get domain of feature value

        Returns:
            tuple: domain
        """
        return None

    @staticmethod
    def get_id_string(feature_id, value_id):
        """Get string for id

        Args:
            feature_id (str): name of feature
            value_id (object): feature value object

        Returns:
            str: id string
        """
        if value_id is not None:
            return '{0}:{1}'.format(feature_id, value_id)
        else:
            return feature_id


class IntFeature(Feature):
    """Base class for inter type feature

    Attributes:
        id (str): string id
        shape(tuple): shape of feature vector
    """
    dtype = int
    default_val = 0
    domain_min = None
    domain_max = None

    def get_domain(self):
        """Get domain (min, max) of feature value

        Returns:
            tuple: (min, max) of domain
        """
        return self.domain_min, self.domain_max


# -----------------------------------------------------------------------------
# FeatureSet: a set of features of molecule
# -----------------------------------------------------------------------------

class FeatureSet(object):
    """A set of features. Features extracted by a feature extractor are grouped in a feature set.

    Attributes:
        id (str): id of a feature set
        features (list): a list of feature objects
        feature_id (str): base id of features in a feature set
        extractor(FeatureExtractor, optional): FeatureExtractor object. Defaults to None.
        id_map (dict): a map from feature id to feature object
        dtype (dtype): data type of feature
        domain (tuple): a tuple of min/max value of a feature
    """

    def __init__(self, id, feature_id, extractor=None, features=None):
        """Constructor of FeatureSet class.

        Args:
            id (str): id of a feature set
            feature_id (str): base id of features in a feature set
            extractor(FeatureExtractor, optional): FeatureExtractor object. Defaults to None.
            features (list): a list of initial features
        """
        self.id = id
        self.feature_id = feature_id
        self.extractor = extractor
        self.features = []
        self.id_map = dict()
        self.dtype = None
        self.domain = None
        if features is not None:
            for f in features:
                self.add_feature(f)

    def get_id(self):
        """Get id of a feature set.

        Returns:
            str: id of a feature set
        """
        return self.id

    def get_feature_id(self):
        """Get base id of feature in a feature set.

        Returns:
            str: base id of a feature
        """
        return self.feature_id

    def get_extractor(self):
        """Get feature extractor for a feature set

        Returns:
            FeatureExtractor: feature extractor
        """
        return self.extractor

    def is_online_update(self):
        """If the extractor update features in the structure generation or not

        Returns:
            bool: ture if online update
        """
        return self.extractor is not None and self.extractor.is_online_update()

    def get_dtype(self):
        """Get data type of a feature

        Returns:
            type: data type
        """
        return self.dtype

    def get_domain(self):
        """Get domain of feature value

        Returns:
            tuple: domain
        """
        return self.domain

    def has_feature(self, id):
        """Check if a feature is in a feature set.

        Args:
            id (str): feature id

        Returns:
            bool: true if a feature is in a feature set
        """
        return id in self.id_map

    def add_feature(self, feature):
        """Add a feature to a feature set.

        Args:
            feature (FeatureExtraction.Feature): a new feature to add
        """
        if self.dtype is None:
            self.dtype = feature.get_dtype()
        if self.domain is None:
            self.domain = feature.get_domain()
        self.features.append(feature)
        self.id_map[feature.id] = feature

    def get_feature(self, id):
        """Get a feature by its id.

        Args:
            id (str): id of a feature

        Returns:
            FeatureExtraction.Feature: a feature object
        """
        return self.id_map[id]

    def get_feature_list(self):
        """Get a list of features in a feature set.

        Returns:
            list: a list of features
        """
        return self.features

    def sort_feature(self):
        """Sort features by their id.
        """
        self.features.sort(key=lambda x: x.id)

    def get_size(self):
        """Get the size of feature set.

        Returns:
            int: size of a feature set
        """
        return len(self.features)

    def get_vector_size(self):
        """Get the size of feature vector.

        Returns:
            int: size of a feature vector
        """
        vector_size = 0
        for f in self.features:
            vector_size += f.get_vector_size()
        return vector_size

    def get_header_list(self):
        """Get a header list of a feature vector in a csv file.

        Returns:
            list: a list of headers
        """
        header_list = []
        for feature in self.features:
            header_list.extend(feature.get_header_list())
        return header_list

    def extract_features(self, mols):
        """Extract features from molecules

        Args:
            mols (list): a list of molecules
        """
        if self.extractor is not None:
            self.extractor.extract_features(mols)

    def make_feature_vector(self, mols, readable=False):
        """Make a feature vector from a set of features.

        Args:
            mols (list): a list of molecules
            readable (bool, optional): flag for value in readable form. Default to False.

        Return:
            DataFrame: a matrix of molecules and feature vectors
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('make features vector: mols=%d features=%d', len(mols), self.get_vector_size())
        # make a feature vector by counting features in a molecule
        if readable:
            feature_vector = np.zeros((len(mols), len(self.features)), dtype=object)
            for (i, mol) in enumerate(mols):
                for (j, feature) in enumerate(self.features):
                    if mol.has_feature(feature):
                        value = mol.get_feature(feature)
                        feature_vector[i][j] = self.to_readable_string(feature, value, self.dtype)
                    else:
                        feature_vector[i][j] = ''
            headers = []
            for feature in self.features:
                headers.append(feature.get_id())
            df_index = [m.id for m in mols]
            dataframe = pd.DataFrame(data=feature_vector, index=df_index, columns=headers)
            return dataframe
        else:
            feature_vector = np.zeros((len(mols), self.get_vector_size()), dtype=self.dtype)
            for (i, mol) in enumerate(mols):
                index = 0
                for feature in self.features:
                    if mol.has_feature(feature):
                        value = mol.get_feature(feature)
                        self.copy_values(feature_vector[i], index, feature, value)
                        index += feature.get_vector_size()
                    else:
                        feature_vector[i][index:index+feature.get_vector_size()] = feature.get_default_value()
                        index += feature.get_vector_size()
            df_index = [m.id for m in mols]
            dataframe = pd.DataFrame(data=feature_vector, index=df_index, columns=self.get_header_list())
            return dataframe

    def copy_values(self, vector, index, feature, feature_value):
        fvalue_list = feature.get_shaped_value(feature_value)
        for (shape, value, default_value) in fvalue_list:
            index = self.copy_values0(vector, index, shape, value, default_value)

    def copy_values0(self, vector, index, shape, value, default_value):
        if len(shape) == 0:
            vector[index] = value
            return index + 1
        elif len(shape) == 1:
            for i in range(min(value.size, shape[0])):
                vector[index + i] = value[i] if value[i] == value[i] else default_value
            vector[index+value.size:shape[0]] = default_value
            return index + shape[0]
        else:
            vector_size = 1
            for i in range(len(shape)-1):
                vector_size *= shape[i+1]
            for i in range(min(value.shape[0], shape[0])):
                index = self.copy_values0(vector, index, shape[1:], value[i], default_value)
            for i in range(min(value.shape[0], shape[0]), shape[0]):
                vector[index:index+vector_size] = default_value
                index += vector_size
            return index

    def to_readable_string(self, feature, value, dtype):
        if len(feature.get_shape()) == 0:
            return feature.get_formatted_value(value)
        elif len(value.shape) == 1:
            str = '['
            if dtype == np.float:
                for i in range(value.size):
                    str += '{0:.3g},'.format(value[i])
            else:
                for i in range(value.size):
                    str += '{0},'.format(value[i])
            return str.rstrip(',')+']'
        else:
            str = '['
            for i in range(value.shape[0]):
                str += self.to_readable_string(feature, value[i], dtype)+','
            return str.rstrip(',')+']'

    def to_string(self):
        """Get a string representation of a feature set (feature set name: [list of feature names]).
        """
        return '{0}:{1}'.format(self.id, [f.id for f in self.features])

    def print_features(self):
        """Print a string representation of a feature set.
        """
        print(self.to_string())


class MergedFeatureSet(FeatureSet):
    """a feature set created by merging existing feature sets.

    Attributes:
        id (str): id of a feature set
        features (list): a list of feature objects
        id_map (dict): a map from feature id to feature object
        features_list (list); a list of merged feature sets
    """

    def __init__(self, features_list):
        """Constructor of MergedFeatureSet class.

        Args:
            features_list (list): a list of feature sets to merge
        """
        super().__init__(self.__get_merged_id(features_list), '')
        self.features_list = features_list
        # add features
        for fs in features_list:
            for f in fs.get_feature_list():
                self.add_feature(f)
        # make slice map
        self.features_slice_map = dict()
        self.vector_slice_map = dict()
        features_size = 0
        vector_size = 0
        for fs in features_list:
            self.features_slice_map[fs.id] = slice(features_size, features_size + fs.get_size())
            self.vector_slice_map[fs.id] = slice(vector_size, vector_size + fs.get_vector_size())
            features_size += fs.get_size()
            vector_size += fs.get_vector_size()

    def get_features_list(self):
        """Get a list of feature sets

        Returns:
            list: a list of feature sets
        """
        return self.features_list

    def get_features_slice_map(self):
        """Get a mapping of features id and a slice of features list

        Returns:
            dict: a mapping of slices
        """
        return self.features_slice_map

    def get_features_slice(self, features):
        """Get a slice of features in a features list

        Args:
            features (FeatureSet): a feature set

        Returns:
            slice: slice of a feature vector
        """
        if features.id in self.features_slice_map:
            return self.features_slice_map[features.id]
        else:
            return None

    def get_feature_vector_slice_map(self):
        """Get a mapping of features id and a slice of feature vector

        Returns:
            dict: a mapping of slices
        """
        return self.vector_slice_map

    def get_feature_vector_slice(self, features):
        """Get a slice of features in a feature vector

        Args:
            features (FeatureSet): a feature set

        Returns:
            slice: slice of a feature vector
        """
        if features.id in self.vector_slice_map:
            return self.vector_slice_map[features.id]
        else:
            return None

    def get_selected_feature_vector_slice_map(self, selection_mask):
        """Get a mapping of features id and a slice of a selected feature vector

        Args:
            selection_mask (list): list of flag of feature value selection.

        Returns:
            dict: a mapping of features id and a slice of a selected feature vector
        """
        selected_vector_slice = dict()
        feature_vector_size = 0
        selected_vector_size = 0
        for features in self.features_list:
            if selection_mask is None:
                vector_size = features.get_vector_size()
                selected_vector_slice[features.id] = slice(selected_vector_size,
                                                           selected_vector_size + vector_size)
            else:
                vector_size = 0
                index = 0
                for feature in features.get_feature_list():
                    for idx in range(feature.get_vector_size()):
                        if selection_mask[feature_vector_size+index+idx]:
                            vector_size += 1
                    index += feature.get_vector_size()
                selected_vector_slice[features.id] = slice(selected_vector_size,
                                                           selected_vector_size+vector_size)
            selected_vector_size += vector_size
            feature_vector_size += features.get_vector_size()
        return selected_vector_slice

    def extract_features(self, mols):
        """Extract features from molecules

        Args:
            mols (list): a list of molecules
        """
        for features in self.features_list:
            features.extractor.extract_features(mols)

    def make_feature_vector(self, mols, readable=False):
        """Make a feature vector from a set of features.

        Args:
            mols (list): a list of molecules
            readable (bool, optional): flag for value in readable form. Default to False.

        Return:
            DataFrame: a matrix of molecules and feature vectors
        """
        df_index = [m.id for m in mols]
        df = pd.DataFrame(index=df_index)
        for features in self.features_list:
            df = df.join(features.make_feature_vector(mols, readable=readable))
        return df

    @staticmethod
    def __get_merged_id(features_list):
        merged_id = '|'
        for fs in features_list:
            merged_id += fs.id+'|'
        return merged_id


# -----------------------------------------------------------------------------
# FeatureExtractor: extractor of features from molecule
# -----------------------------------------------------------------------------

class FeatureExtractor(object):
    """Base class of feature extraction.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor

    Note:
        Feature extractor is assumed to apply to molecules managed in moldata by default
    """

    id_string = ''
    """str: Base name of id string of an extractor class"""

    feature_id_string = ''
    """str: Base name of id string of feature extracted by an extractor class"""

    online_update = False
    """bool: flag of online update of features in the structure generation"""

    def __init__(self, moldata):
        """Constructor of FeatureExtractor class.

        Args:
            moldata (MolData): a molecule data management object
        """
        self.moldata = moldata
        self.params = dict()

    def get_id(self):
        """Get id of an extractor.

        Returns:
            str: id string
        """
        id = self.id_string
        if len(self.params) > 0:
            id += ':'
            for key in sorted(self.params.keys()):
                id += '{0}={1} '.format(key, self.params[key])
        return id.rstrip()

    def get_params(self):
        """Get parameters of an extractor.

        Returns:
            dict: parameters of an extractor
        """
        return self.params

    def is_online_update(self):
        """If the extractor update features in the structure generation or not

        Returns:
            bool: ture if online update
        """
        return self.online_update

    @classmethod
    def get_feature_id(cls):
        """Get id of feature extracted by an extractor

        Returns:
            str: id string
        """
        return cls.feature_id_string

    def extract(self):
        """Extract features from molecules managed in moldata.

        Returns:
            FeatureSet, DataFrame, list: a set of extracted features,
            a list of mask of valid molecules
        """
        features, feature_mask = self.extract_features(self.moldata.mols)
        features.sort_feature()
        return features, feature_mask

    def extract_features(self, mols):
        """Extract features from a list of molecules.

        Args:
            mols (list): a list of molecule

        Returns:
            FeatureSet, list: a set of features, a list of mask of valid molecules
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('extract features: {0}'.format(self.get_id()))
        features = FeatureSet(self.get_id(), self.get_feature_id(), extractor=self)
        feature_mask = []
        for mol in mols:
            if logger.isEnabledFor(logging.INFO):
                logger.info('feature: %s: mol:%s %s', self.get_id(), mol.get_id(), mol.get_smiles())
            feature_map = self.extract_mol_features(mol, features)
            if feature_map is not None:
                # set feature to mol
                for feature, value in feature_map.items():
                    mol.set_feature(feature, value)
                feature_mask.append(True)
            else:
                logger.error('failed to extract feature: %s: mol:%s %s',
                             self.get_id(), mol.get_id(), mol.get_smiles())
                feature_mask.append(False)
        if all(feature_mask):
            feature_mask = None
        if logger.isEnabledFor(logging.INFO):
            logger.info('%s: extracted %d unique features',
                        self.get_id(), features.get_size())
        return features, feature_mask

    def extract_mol_features(self, mol, features=None):
        """Extract features from a molecule. A newly found features are added to a feature set.
        This method should be implemented in subclasses.

        Args:
            mol (Molecule): a molecule to extract features from
            features (FeatureSet, optional): a set of features already extracted from other molecules. Defaults to None.

        Returns:
            dict: number of counting
        """
        raise NotImplementedError('FeatureExtractor:extract_mol_features()')

    def update_feature_value(self, graph, new_vertex, feature_list, node, updated):
        """Update feature values due to new vertex incrementally.

        This is a default feature value updater using extract_mol_features().
        Considering the performance, this should be overwritten by incremental update method.

        Args:
            graph (AtomGraph): graph of a molecule
            new_vertex (AtomVertex): a vertex newly added to a graph
            feature_list (list): a list of feature to update
            node (AtomGraphNode): search node of structure generator
            updated (set): a set of updated feature set id
        """
        if self.get_id() in updated:
            return
        feature_values = node.feature_values
        mol = feature_values['amd_tool'].get('mol', None)
        if mol is None:
            mol = SimpleMolecule('online', graph=graph)
            feature_values['amd_tool']['mol'] = mol
        feature_map = self.extract_mol_features(mol)
        if feature_map is None:
            logger.error('failed to extract online feature: %s: smiles:%s',
                         self.get_id(), mol.get_smiles())
        else:
            for feature in feature_list:
                if feature.id in feature_map:
                    value = feature_map[feature.id]
                else:
                    value = np.zeros(shape=feature.shape)
                feature_values[self.get_id()][feature.id] = value
                mol.set_feature(feature, value)


class StructureCounting(FeatureExtractor):
    """Base class of feature extractor for counting occurrences of a certain structure
    in a molecule.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
    """

    class Feature(IntFeature):
        """Feature of structure counting
        """
        domain_min = 0
        domain_max = None

    def __init__(self, moldata):
        """Constructor of StructureCounting class.

        Args:
            moldata (MolData): a molecule data management object
        """
        super().__init__(moldata)


class HeavyAtomExtractor(StructureCounting):
    """Feature extractor for counting the number of heavy atoms.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
    """

    id_string = 'heavy_atom'
    """str: Base name of id string for HeavyAtomExtractor"""

    feature_id_string = 'atom'
    """str: Base name of id string for feature of HeavyAtomExtractor"""

    online_update = True
    """bool: flag of online update of features in the structure generation"""

    class Feature(StructureCounting.Feature):
        """Feature of molecule by heavy atom count

        Attributes:
            id (str): id of heavy atom
            shape (tuple): shape of feature vector
            atom (str): atom symbol
            charge (int): charge
            valence(int): valence of atom
        """

        def __init__(self, atom):
            """Constructor of FeatureByAtom class.

            Args:
                atom (Atom): Atom object of rdkit
            """
            symbol = ChemVertex.atom_to_symbol(atom.GetSymbol(), atom.GetFormalCharge())
            super().__init__(HeavyAtomExtractor.get_feature_id(), symbol)
            self.symbol = symbol
            self.atom = atom.GetSymbol()
            self.charge = atom.GetFormalCharge()
            self.valence = atom.GetTotalValence()

        def get_index(self):
            """Get index of dictionary.

            Returns:
                str, int: index
            """
            return self.symbol

        def get_symbol(self):
            """Get atom symbol.

            Returns:
                str: atom symbol
            """
            return self.symbol

        def get_atom(self):
            """Get atom name.

            Returns:
                str: atom name
            """
            return self.atom

        def get_charge(self):
            """Get charge.

            Returns:
                int: charge
            """
            return self.charge

        def get_valence(self):
            """Get valence.

            Returns:
                int: valence
            """
            return self.valence

    def __init__(self, moldata):
        """Constructor of HeavyAtomExtractor class.

        Args:
            moldata (MolData): a molecule data management object
        """
        super().__init__(moldata)

    def extract_mol_features(self, mol, features=None):
        """Extract the number of heavy atoms from a molecule. A newly found atoms are added
        to a feature set if it is given.

        Args:
            mol (Molecule): a molecule to extract features from
            features (FeatureSet, optional): a set of features already extracted from other molecules. Defaults to None.

        Returns:
            int: number of heavy atoms
        """
        if not isinstance(mol, SimpleMolecule):
            return None
        atoms = Counter()
        rdkit_atom = dict()
        # need update property cache to avoid explicitValence error by rdkit
        mol.get_mol().UpdatePropertyCache()
        # count atoms
        for atom_index in range(mol.get_mol().GetNumAtoms()):
            atom = mol.get_mol().GetAtomWithIdx(atom_index)
            if atom.GetAtomicNum() > 1:  # skip atom 'H' and '*'
                symbol = ChemVertex.atom_to_symbol(atom.GetSymbol(), atom.GetFormalCharge())
                atoms[symbol] += 1
                if symbol not in rdkit_atom:
                    rdkit_atom[symbol] = atom
                elif rdkit_atom[symbol].GetTotalValence() < atom.GetTotalValence():
                    rdkit_atom[symbol] = atom
        # set atom feature
        feature_count = Counter()
        for atom, count in atoms.items():
            feature = self.Feature(rdkit_atom[atom])
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('feature %s:%s', self.get_id(), feature.id)
            if features is not None:
                # update feature set
                if not features.has_feature(feature.id):
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('new feature %s:%s', self.get_id(), feature.id)
                    features.add_feature(feature)
                else:
                    # update valence of existing feature
                    if features.get_feature(feature.id).get_valence() < feature.get_valence():
                        features.get_feature(feature.id).valence = feature.get_valence()
                    feature = features.get_feature(feature.id)
            feature_count[feature] = count
        return feature_count

    def update_feature_value(self, graph, new_vertex, feature_list, node, updated):
        """Update feature values due to new vertex incrementally.

        Args:
            graph (AtomGraph): graph of a molecule
            new_vertex (AtomVertex): a vertex newly added to a graph
            feature_list (list): a list of feature to update
            node (AtomGraphNode): search node of structure generator
            updated (set): a set of updated feature set id
        """
        if self.get_id() in updated:
            return
        feature_values = node.feature_values
        mol = feature_values['amd_tool'].get('mol', None)
        if mol is None:
            mol = SimpleMolecule('online', graph=graph)
            feature_values['amd_tool']['mol'] = mol
        # copy counter from built-in counter of search node
        for feature in feature_list:
            value = node.atom_count[feature.get_index()]
            feature_values[self.get_id()][feature.id] = value
            mol.set_feature(feature, value)


class RingExtractor(StructureCounting):
    """Feature extractor for counting the number of rings.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
    """

    id_string = 'ring'
    """str: Base name of id string for RingExtractor"""

    feature_id_string = 'ring'
    """str: Base name of id string for feature extracted by RingExtractor"""

    online_update = True
    """bool: flag of online update of features in the structure generation"""

    use_rdkit = True

    class Feature(StructureCounting.Feature):
        """Feature of molecule by ring count.

        Attributes:
            id (str): id of ring ('ring:'+ring size)
            shape (tuple): tuple of feature vector
            ring_size (int): size of a ring
        """

        def __init__(self, ring_size):
            """Constructor of FeatureByRing class.

            Args:
                ring_size (int): size of a ring
            """
            super().__init__(RingExtractor.get_feature_id(), ring_size)
            self.ring_size = ring_size

        def get_index(self):
            """Get index of dictionary.

            Returns:
                str, int: index
            """
            return self.ring_size

        def get_ring_size(self):
            """Get ring size.

            Returns:
                int: ring size
            """
            return self.ring_size

    def __init__(self, moldata):
        """Constructor of RingExtractor class.

        Args:
            moldata (MolData): a molecule data management object
        """
        super().__init__(moldata)

    def extract_mol_features(self, mol, features=None):
        """Extract the number of rings from a molecule. A newly found rings are added
        to a feature set if it is given.

        Args:
            mol (Molecule): a molecule to extract features from
            features (FeatureSet, optional): a set of features already extracted from other molecules. Defaults to None.

        Returns:
            int: number of rings
        """
        if not isinstance(mol, SimpleMolecule):
            return None
        rings = Counter()
        # count rings and aromatic rings
        if self.use_rdkit:
            atom_rings = mol.get_mol().GetRingInfo().AtomRings()
            for ring in atom_rings:
                rings[len(ring)] += 1
        else:
            rings, aromaics = mol.get_graph().count_rings_sssr()
        # set ring feature
        feature_count = Counter()
        for ring, count in rings.items():
            feature = self.Feature(ring)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('feature %s:%s', self.get_id(), feature.id)
            if features is not None:
                # update feature set
                if not features.has_feature(feature.id):
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('new feature %s:%s', self.get_id(), feature.id)
                    features.add_feature(feature)
                else:
                    feature = features.get_feature(feature.id)
            feature_count[feature] = count
        return feature_count

    def update_feature_value(self, graph, new_vertex, feature_list, node, updated):
        """Update feature values due to new vertex incrementally.

        Args:
            graph (AtomGraph): graph of a molecule
            new_vertex (AtomVertex): a vertex newly added to a graph
            feature_list (list): a list of feature to update
            node (AtomGraphNode): search node of structure generator
            updated (set): a set of updated feature set id
        """
        if self.get_id() in updated:
            return
        feature_values = node.feature_values
        mol = feature_values['amd_tool'].get('mol', None)
        if mol is None:
            mol = SimpleMolecule('online', graph=graph)
            feature_values['amd_tool']['mol'] = mol
        # copy counter from built-in counter of search node
        for feature in feature_list:
            value = node.num_ring_count[feature.get_index()]
            feature_values[self.get_id()][feature.id] = value
            mol.set_feature(feature, value)


class AromaticRingExtractor(StructureCounting):
    """Feature extractor for counting the number of aromatic rings.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
    """

    id_string = 'aromatic_ring'
    """str: Base name of id string for AromaticRingExtractor"""

    feature_id_string = 'aring'
    """str: Base name of id string for feature extracted by AromaticRingExtractor"""

    online_update = True
    """bool: flag of online update of features in the structure generation"""

    use_rdkit = True

    class Feature(StructureCounting.Feature):
        """Feature of molecule by aromatic ring count.

        Attributes:
            id (str): id of aromatic ring ('aring:'+ring size)
            shape (tuple): shape of feature vector
            ring size (int): size of an aromatic ring
        """

        def __init__(self, ring_size):
            """Constructor of FeatureByAromaticRing class.

            Args:
                ring size (int): size of an aromatic ring
            """
            super().__init__(AromaticRingExtractor.get_feature_id(), ring_size)
            self.ring_size = ring_size

        def get_index(self):
            """Get index of dictionary.

            Returns:
                str, int: index
            """
            return self.ring_size

        def get_ring_size(self):
            """Get ring size.

            Returns:
                int: aromatic ring size
            """
            return self.ring_size

    def __init__(self, moldata):
        """Constructor of AromaticRingExtractor class.

        Args:
            moldata (MolData): a molecule data management object
        """
        super().__init__(moldata)

    def extract_mol_features(self, mol, features=None):
        """Extract the number of aromatic rings from a molecule. A newly found aromatic rings are added
        to a feature set if it is given.

        Args:
            mol (Molecule): a molecule to extract features from
            features (FeatureSet, optional): a set of features already extracted from other molecules. Defaults to None.

        Returns:
            int: number of aromatic rings
        """
        if not isinstance(mol, SimpleMolecule):
            return None
        aromatics = Counter()
        # count rings and aromatic rings
        if self.use_rdkit:
            atom_rings = mol.get_mol().GetRingInfo().AtomRings()
            for ring in atom_rings:
                # check aromatic ring
                if all(mol.get_mol().GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                    aromatics[len(ring)] += 1
        else:
            rings, aromatics = mol.get_graph().count_rings_sssr()
        # set aromatic ring feature
        feature_count = Counter()
        for ring, count in aromatics.items():
            feature = self.Feature(ring)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('feature %s:%s', self.get_id(), feature.id)
            if features is not None:
                # update feature set
                if not features.has_feature(feature.id):
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('new feature %s:%s', self.get_id(), feature.id)
                    features.add_feature(feature)
                else:
                    feature = features.get_feature(feature.id)
            feature_count[feature] = count
        return feature_count

    def update_feature_value(self, graph, new_vertex, feature_list, node, updated):
        """Update feature values due to new vertex incrementally.

        Args:
            graph (AtomGraph): graph of a molecule
            new_vertex (AtomVertex): a vertex newly added to a graph
            feature_list (list): a list of feature to update
            node (AtomGraphNode): search node of structure generator
            updated (set): a set of updated feature set id
        """
        if self.get_id() in updated:
            return
        feature_values = node.feature_values
        mol = feature_values['amd_tool'].get('mol', None)
        if mol is None:
            mol = SimpleMolecule('online', graph=graph)
            feature_values['amd_tool']['mol'] = mol
        # copy counter from built-in counter of search node
        for feature in feature_list:
            value = node.num_aromatic_count[feature.get_index()]
            feature_values[self.get_id()][feature.id] = value
            mol.set_feature(feature, value)


class StructureExtractor(StructureCounting):
    """Base class of feature extractor for counting the occurrence of structure

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
    """

    class Feature(StructureCounting.Feature):
        """Base class for features of molecule by sub-structure count.

        Attributes:
            id (str): id of sub-structure
            shape (tuple): shape of feature vector
            mol (Mol): molecule object of rdkit
            graph (AtomGraph): graph of a molecular structure
            fragment (ChemFragment): fragment of molecule generator

        Note:
            graph and fragment objects are generated from mol when necessary
        """

        def __init__(self, id, value_id, mol):
            """Constructor of FeatureByStructure class.

            Args:
                mol (Mol): molecule object of rdkit
            """
            super().__init__(id, value_id)
            self.mol = mol
            self.graph = None
            self.fragment = None

        def get_index(self):
            """Get index of dictionary.

            Returns:
                str, int: index
            """
            return self.fragment

        def get_graph(self):
            """Generate a graph of a molecule, and cache to a member variable.

            Returns:
                AtomGraph: graph object of a molecule
            """
            if self.graph is None:
                self.graph = AtomGraph(mol=self.mol)
            return self.graph

        def get_fragment(self):
            """Generate a fragment for a molecule generator, and cache to a member variable
            fragment object is used to count the occurrence of sub-structure in a graph.

            Returns:
                ChemFragment: fragment object of a molecule
            """
            if self.fragment is None:
                self.fragment = ChemFragment(self.get_graph())
            return self.fragment

    def expand_mol(self, mol):
        """Expand a molecule by replacing dummy atoms

        Args:
            mol (Molecule): a molecule

        Returns:
            Mol: rdkit mol of expanded molecule
        """
        if isinstance(mol, GeneratedMolecule):
            return mol.get_expand_mol()
        elif isinstance(mol, SimpleMolecule):
            return mol.get_mol()
        else:
            return None


class FingerPrintStructureExtractor(StructureExtractor):
    """Feature extractor for counting the occurrence of fingerprint structure of given radius.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
        radius (int): radius of fingerprint structure
    """

    id_string = 'finger_print_structure'
    """str: Base name of id string for FingerPrintStructureExtractor"""

    feature_id_string = 'fp'
    """str: Base name of id string for feature extracted by FingerPrintStructureExtractor"""

    online_update = True
    """bool: flag of online update of features in the structure generation"""

    class Feature(StructureExtractor.Feature):
        """Feature of molecule by finger-print sub-structure.

        Attributes:
            id (str): id of a sub-structure (SMILES representation)
            shape (tuple): shape of feature vector
            mol (Mol): molecule object of rdkit
            graph (AtomGraph): graph of a molecular structure
            fragment (ChemFragment): fragment of molecule generator
            root_atom (str): root atom symbol of fingerprint sub-structure
            radius (int): radius of fingerprint sub-structure
        """

        def __init__(self, id, mol, root_index, radius):
            """Constructor of FeatureByFingerPrint class.

            Args:
                mol (Mol): molecule object of rdkit
                root_index (int): an index of root atom
                radius (int): radius of fingerprint
            """
            super().__init__(id, self.get_value_string(mol, root_index, radius), mol)
            # mark root atom by atom property
            self.root_index = root_index
            root_atom = self.mol.GetAtomWithIdx(root_index)
            root_atom.SetUnsignedProp('root', 0)
            self.root_atom = ChemVertex.atom_to_symbol(root_atom.GetSymbol(),
                                                       root_atom.GetFormalCharge())
            self.radius = radius
            # mark root group atom by atom property
            root_group = set()
            old_root_group = set()
            root_group.add(root_index)
            old_root_group.add(root_index)
            for depth in range(1, radius):
                new_root_group = set()
                for atom_index in old_root_group:
                    for bond in mol.GetAtomWithIdx(atom_index).GetBonds():
                        if bond.GetBeginAtomIdx() != atom_index:
                            new_atom = bond.GetBeginAtom()
                        else:
                            new_atom = bond.GetEndAtom()
                        if new_atom.GetIdx() not in root_group:
                            root_group.add(new_atom.GetIdx())
                            new_root_group.add(new_atom.GetIdx())
                            new_atom.SetUnsignedProp('root', depth)
                old_root_group = new_root_group

        def get_root_atom(self):
            """Get root atom symbol.

            Returns:
                string: atom symbol
            """
            return self.root_atom

        def is_aromatic_root_atom(self):
            """Get if root atom is aromatic.

            Returns:
                bool: True if atom is aromatic
            """
            return self.mol.GetAtomWithIdx(self.root_index).GetIsAromatic()

        def is_terminating_root_atom(self, bond_type):
            """Get if root atom is terminating atom with given bond type

            Returns:
                bool: Ture if atom is terminating
            """
            root_atom = self.mol.GetAtomWithIdx(self.root_index)
            if root_atom.GetDegree() != 1:
                return False
            bond = root_atom.GetBonds()[0]
            if bond.GetBondType() != bond_type:
                return False
            return True

        @staticmethod
        def get_value_string(mol, root_index, radius):
            """Get a string as a SMILES of a molecule and root atom index.

            Args:
                mol (Mol): rdkit Mol object
                root_index (int): index of root atom
                radius (int): radius

            Returns:
                str: id string
            """
            root_atom = mol.GetAtomWithIdx(root_index)
            root_symbol = root_atom.GetSymbol()
            if root_atom.GetIsAromatic():
                root_symbol = root_symbol.lower()
            root_num = root_atom.GetAtomicNum()
            root_atom.SetAtomicNum(0)
            id_string = 'r{0}[{1}]:{2}'.format(radius, root_symbol, Chem.MolToSmiles(mol))
            root_atom.SetAtomicNum(root_num)
            return id_string

    def __init__(self, moldata, radius=1):
        """Constructor of FingerPrintStructureExtractor class.

        Args:
            moldata (MolData): a molecule data management object
            radius (int, optional): radius of fingerprint structure. Defaults to 1.
        """
        super().__init__(moldata)
        self.radius = radius
        self.params['radius'] = radius

    def set_params(self, radius=None):
        """Set parameter to feature extractor.

        Args:
            radius (int, optional): radius of fingerprint structure. Defaults to None.
        """
        if radius is not None:
            self.radius = radius
            self.params['radius'] = radius

    def extract_mol_features(self, mol, features=None):
        """Extract the number of fingerprint structures of given radius from a molecule.
        A newly found sub-structures are added to a feature set if it is given.

        Args:
            mol (Molecule): a molecule to extract features from
            features (FeatureSet, optional): a set of features already extracted from other molecules. Defaults to None.

        Returns:
            int: number of fingerprint structures
        """
        # expand dummy atoms
        expand_mol = self.expand_mol(mol)
        if expand_mol is None:
            return None

        num_atom = 0
        feature_map = dict()
        for atom_index in range(0, expand_mol.GetNumAtoms()):
            path = Chem.FindAtomEnvironmentOfRadiusN(expand_mol, self.radius, atom_index)
            if len(path) == 0:
                continue
            num_atom += 1
            sub_mol, root_index = self.path_to_submol(expand_mol, path, atom_index, self.radius)
            feature = self.Feature(self.get_feature_id(), sub_mol, root_index, self.radius)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('feature %s[%d]:%s', self.get_id(), atom_index, feature.id)
            if feature.id not in feature_map:
                feature_map[feature.id] = feature
                if features is not None:
                    if not features.has_feature(feature.id):
                        if logger.isEnabledFor(logging.INFO):
                            logger.info('new feature %s[%d]:%s', self.get_id(), atom_index, feature.id)
                        features.add_feature(feature)

        # count feature
        graph = AtomGraph(mol=expand_mol)
        labeling = ChemGraphLabeling(graph.vertices)
        feature_count = Counter()
        total_count = 0
        for feature in feature_map.values():
            fragment = feature.get_fragment()
            count = fragment.count_fragment_graph(graph, labeling)
            if count == 0:
                logger.error('zero %s feature count!!: mol:%s %s feature:%s',
                             self.get_id(), mol.get_id(), mol.get_smiles(), feature.id)
                # return None
                feature_count[feature] = 0
            else:
                feature_count[feature] = count
                total_count += count
        if total_count < num_atom:
            logger.error('feature count %d is not the number of atoms %d: mol:%s %s',
                         total_count, num_atom, mol.get_id(), mol.get_smiles())
        return feature_count

    @staticmethod
    def path_to_submol(mol, path, root_index, radius):
        """make a fingerprint sub-structure of a molecule from given list of bond index.

        Chem.PathToSubmol() has a problem of adding a bond between atoms at the distance of radius.

        Args:
            mol (Mol): a molecule
            path (array): an array of bond indices
            root_index (int): index of a root atom of a fingerprint
            radius (int): radius of a fingerprint

        Returns:
            Mol, int: a new molecule, a root atom index of a new molecule
        """
        atom_radius_map = dict()
        atom_index_map = dict()
        mw = Chem.RWMol()
        # add root atom
        atom = mol.GetAtomWithIdx(root_index)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        # new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_atom_index = mw.AddAtom(new_atom)
        atom_index_map[atom.GetIdx()] = new_atom_index
        atom_radius_map[atom.GetIdx()] = 0
        for bond_index in path:
            bond = mol.GetBondWithIdx(bond_index)
            satom = bond.GetBeginAtom()
            eatom = bond.GetEndAtom()
            if satom.GetIdx() not in atom_index_map:
                new_atom = Chem.Atom(satom.GetSymbol())
                new_atom.SetFormalCharge(satom.GetFormalCharge())
                # new_atom.SetNumExplicitHs(satom.GetNumExplicitHs())
                new_atom_index = mw.AddAtom(new_atom)
                atom_index_map[satom.GetIdx()] = new_atom_index
                atom_radius_map[satom.GetIdx()] = atom_radius_map[eatom.GetIdx()] + 1
            if eatom.GetIdx() not in atom_index_map:
                new_atom = Chem.Atom(eatom.GetSymbol())
                new_atom.SetFormalCharge(eatom.GetFormalCharge())
                # new_atom.SetNumExplicitHs(eatom.GetNumExplicitHs())
                new_atom_index = mw.AddAtom(new_atom)
                atom_index_map[eatom.GetIdx()] = new_atom_index
                atom_radius_map[eatom.GetIdx()] = atom_radius_map[satom.GetIdx()] + 1
            # add bond between atoms within radius
            if atom_radius_map[satom.GetIdx()] < radius or \
                    atom_radius_map[eatom.GetIdx()] < radius:
                mw.AddBond(atom_index_map[satom.GetIdx()],
                           atom_index_map[eatom.GetIdx()],
                           bond.GetBondType())
        submol = mw.GetMol()
        try:
            submol.UpdatePropertyCache()
        except ValueError as e:
            logger.error('path_to_submol:mol:%s, submol:%s, error:%s',
                         Chem.MolToSmiles(mol), Chem.MolToSmiles(submol), e)
        return submol, atom_index_map[root_index]

    def update_feature_value(self, graph, new_vertex, feature_list, node, updated):
        """Update feature values due to new vertex incrementally.

        Args:
            graph (AtomGraph): graph of a molecule
            new_vertex (AtomVertex): a vertex newly added to a graph
            feature_list (list): a list of feature to update
            node (AtomGraphNode): search node of structure generator
            updated (set): a set of updated feature set id
        """
        if self.get_id() in updated:
            return
        feature_values = node.feature_values
        mol = feature_values['amd_tool'].get('mol', None)
        if mol is None:
            mol = SimpleMolecule('online', graph=graph)
            feature_values['amd_tool']['mol'] = mol
        # copy counter from built-in counter of search node
        for feature in feature_list:
            fragment = feature.get_fragment()
            if fragment in node.fragment_count:
                value = node.fragment_count[fragment]
            else:
                value = fragment.count_fragment_graph(mol.get_graph(),
                                                      labeling=mol.get_graph_labeling())
            if value > 0:
                feature_values[self.get_id()][feature.id] = value
                mol.set_feature(feature, value)


class FeatureOperator(FeatureExtractor):
    """Base class of feature extractor for operating on existing features.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
        features (FeatureSet): target feature set of operator
    """

    online_update = True
    """bool: flag of online update of features in the structure generation"""

    class Feature(Feature):
        """features derived from existing feature by some operator

        Attributes:
            id (str): id of feature
            shape (tuple): shape of feature vector
            feature_dtype (type): data type of result of operator
            default_val (object): default value of a feature
        """

        def __init__(self, feature_id, value_id, dtype, default_val, domain, shape=()):
            """Constructor of FeatureOperator.Feature class.

            Args:
                feature_id (str): id of feature
                value_id (object): value object
                dtype (type): data type
                default_val (object): default value
            """
            super().__init__(feature_id, value_id, shape)
            self.feature_dtype = dtype
            self.default_val = default_val
            self.domain = domain

        def get_dtype(self):
            """Get data type of result of operator

            Returns:
                type: data type
            """
            return self.feature_dtype

        def get_default_value(self):
            """Get default value of feature

            Returns:
                 object: default value
            """
            return self.default_val

        def get_domain(self):
            """Get domain of feature value

            Returns:
                tuple: domain
            """
            return self.domain

        @staticmethod
        def get_id_string(feature_id, value_id):
            """Get string for id

            Args:
                feature_id (str): name of feature
                value_id (object): feature value object

            Returns:
                str: id string
            """
            if value_id is not None:
                return '{0}[{1}]'.format(feature_id, value_id)
            else:
                return feature_id

    def __init__(self, moldata, features):
        """Constructor of FeatureOperator class.

        Args:
            moldata (MolData): a molecule data management object
            features (FeatureSet): target feature set of Operator
        """
        super().__init__(moldata)
        self.features = features

    def get_id(self):
        """Get id of an extractor.

        Returns:
            str: id string
        """
        id = self.features.get_id() + ':' + super().get_id()
        return id

    def get_feature_id(self, feature=None):
        """Get id of feature extracted by an extractor.

        Args:
            feature (Feature, optional): individual feature. Defaults to None

        Returns:
            str: id string
        """
        if feature is None:
            id = self.features.get_feature_id()+':'+super().get_feature_id()
        else:
            id = feature.get_id()+':'+super().get_feature_id()
        return id

    def get_target_features(self):
        """Get target feature set for the operation

        Returns:
            FeatureSet: a feature set
        """
        return self.features

    def is_online_update(self):
        """If the extractor update features in the structure generation or not

        Returns:
            bool: ture if online update
        """
        if self.online_update:
            return self.features.is_online_update()
        else:
            return False

    def extract_features(self, mols):
        """Extract features from a list of molecules.

        Args:
            mols (list): a list of molecule

        Returns:
            FeatureSet, list: a set of features, a list of mask of valid molecules
        """
        if self.features is None:
            logger.error('feature set to apply operator is None')
            return None, []
        return super().extract_features(mols)

    def update_feature_value(self, graph, new_vertex, feature_list, node, updated):
        """Update feature values due to new vertex incrementally.

        This is a default feature value updater using extract_mol_features().
        Considering the performance, this should be overwritten by incremental update method.

        Args:
            graph (AtomGraph): graph of a molecule
            new_vertex (AtomVertex): a vertex newly added to a graph
            feature_list (list): a list of feature to update
            node (AtomGraphNode): search node of structure generation
            updated (set): a set of updated feature set id
        """
        if self.get_id() in updated:
            return
        feature_values = node.feature_values
        # update target feature first
        if self.features is None:
            logger.error('feature set to apply operator is None')
        else:
            self.features.get_extractor().\
                update_feature_value(graph, new_vertex, self.features.get_feature_list(),
                                     node, updated)
        # apply operator to get updated feature value
        mol = feature_values['amd_tool'].get('mol', None)
        if mol is None:
            mol = SimpleMolecule('online', graph=graph)
            feature_values['amd_tool']['mol'] = mol
        feature_map = self.extract_mol_features(mol)
        if feature_map is None:
            logger.error('failed to extract online feature: %s: smiles:%s',
                         self.get_id(), mol.get_smiles())
        else:
            for feature in feature_list:
                if feature.id in feature_map:
                    value = feature_map[feature.id]
                else:
                    value = np.zeros(shape=feature.shape)
                feature_values[self.get_id()][feature.id] = value
                mol.set_feature(feature, value)

    @staticmethod
    def get_operator_name(operator):
        """Get string representation of a function

        Returns:
            str: function name
        """
        if operator.__name__ == '<lambda>':
            m = re.search(r'lambda.*:(.*)', inspect.getsource(operator))
            if m is not None:
                name = m.group(1)
                count_p = 0
                for index, c in enumerate(name):
                    if c in '([':
                        count_p += 1
                    elif c in ')]':
                        count_p -= 1
                    elif count_p == 0 and c in ',\n':
                        name = name[:index]
                        break
                name = name.replace('\n', ' ').strip(' ')
            else:
                name = operator.__name__
        else:
            name = operator.__name__
        return name


class FeatureSumOperator(FeatureOperator):
    """Get a sum of feature values in a feature set.

    Attributes:
        moldata (MolData): a molecule data management object
        params (dict): parameters of an extractor
        features (FeatureSet): target feature set of operator
    """

    id_string = 'sum()'
    """str: Base name of id string of FeatureSumOpeariton"""

    feature_id_string = 'sum'
    """str: Base name of id string of feature extracted by FeatureSumOperator"""

    online_update = True
    """bool: flag of online update of features in the structure generation"""

    class Feature(FeatureOperator.Feature):
        """features derived from existing feature by some operator

        Attributes:
            id (str): id of feature
            shape (tuple): shape of feature vector
            feature_dtype (type): data type of result of operator
            default_val (object): default value of a feature
        """

        def __init__(self, feature_id, value_id, dtype, domain):
            """Constructor of FeatureSumOperator.Feature class.

            Args:
                feature_id (str): id of feature
                value_id (object): value object
                dtype (type): data type
                domain (tuple): domain of feature value
            """
            super().__init__(feature_id, value_id, dtype, 0, domain)

    def __init__(self, moldata, features):
        """Constructor of FeatureSumOperator class.

        Args:
            moldata (MolData): a molecule data management object
            features (FeatureSet): target feature set of operator
        """
        super().__init__(moldata, features)

    def extract_mol_features(self, mol, features=None):
        """Extract the sum of feature values

        Args:
            mol (Molecule): a molecule to extract features from
            features (FeatureSet, optional): a set of features already extracted from other molecules. Defaults to None.

        Returns:
            dict: a mapping of a feature and the sum
        """
        new_feature = self.Feature(self.get_feature_id(), None, self.features.get_dtype(), self.features.get_domain())
        if features is not None:
            if not features.has_feature(new_feature.id):
                features.add_feature(new_feature)
            else:
                new_feature = features.get_feature(new_feature.id)
        feature_sum = {new_feature: 0}
        for feature in self.features.get_feature_list():
            if mol.has_feature(feature):
                feature_sum[new_feature] += mol.get_feature(feature)
        return feature_sum
