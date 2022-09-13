# -*- coding:utf-8 -*-
"""
DataBox.py

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
from rdkit.Chem import PandasTools, Draw

from .Molecule import *
from .FeatureExtraction import *
from .Prediction import *
from .FeatureEstimation import *
from .Generation import *
from .Utility import *

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold

import pickle
import copy
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# MolData: a class for managing molecule data and analytics results
# -----------------------------------------------------------------------------

class MolData(object):
    """Class for managing the analysis of molecules. Staring from registering molecules and their properties,
    MolData object maintains extracted features and regression models applied to a specified property.

    Attributes:
        mol_type (MolType): type of molecule (simple only)
        mol_label (str): label of data set
        mols (list): a list of Molecule objects
        mols_index (dict): a map from Molecule id to index in a list
        mols_mask (list): a list of valid data flag
        anion (MolData): MolData for anion part of molecules
        cation (MolData): MolData for cation part of molecules
        main_ion (MolData): MolData for mian ion part
        sub_ion (MolData): MolData for sub ion part
        properties (PropertySet): a set of properties defined for molecules
        features_index (dict): a map from a feature set to an index in a feature set list
        features_list (list): a list of feature sets
        features_mask_list (list): a list of value mask for feature vectors
        merged_features_index (dict): a map from a merged feature set to an index in a merged feature set list
        merged_features_list (list): a list of merged feature sets
        merged_features_mask_list (list): a list of value mask for merged feature vectors
        regression_model_index (dict): a map from target property and a feature set to an index of a regression model
            in a list
        regression_model_list (dict): a list of regression models for each target property and a feature set
    """

    def __init__(self, mols, properties=PropertySet(), mol_type=MolType.SIMPLE, mol_label=''):
        """Constructor of MolData object.

        Args:
            mols (list): a list of Molecule objects
            properties (PropertySet): a list of properties of molecules. Defaults to empty PropertySet.
            mol_type (MolType, optional): type of Molecule. Defaults to MolType.SIMPLE
            mol_label (str, optional): label of data set. Defaults to ''
        """
        self.mol_type = mol_type
        self.mol_label = mol_label
        self.mols = mols
        self.mols_index = {mol.get_id(): index for index, mol in enumerate(mols)}
        self.mols_mask = None
        self.anion = None
        self.cation = None
        self.main_ion = None
        self.sub_ion = None
        self.properties = properties
        self.features_index = dict()
        self.features_list = []
        self.features_mask_list = []
        self.merged_features_index = dict()
        self.merged_features_list = []
        self.merged_features_mask_list = []
        self.regression_model_index = defaultdict(self.ddict)
        self.regression_model_list = defaultdict(self.dlist)
        self.feature_estimate_index = defaultdict(self.ddict)
        self.feature_estimate_list = defaultdict(self.dlist)

    @staticmethod
    def ddict():
        return defaultdict(dict)

    @staticmethod
    def dlist():
        return defaultdict(list)

    def set_mol_type(self, mol_type):
        """Set molecule type

        Args:
            mol_type (MolType): type of molecule
        """
        self.mol_type = mol_type

    def get_mol_type(self):
        """Get molecule type

        Returns:
            MolType: type of molecule
        """
        return self.mol_type

    def set_mol_label(self, label):
        """Set molecule data label

        Args:
            label (str): data label
        """
        self.mol_label = label

    def get_mol_label(self):
        """Get molecule data label

        Returns:
            str: data label
        """
        return self.mol_label

    def get_mols(self):
        """Get a list of molecules.

        Returns:
            list: a list of molecules
        """
        return self.mols

    def set_mols_mask(self, mask):
        """Set a list of mol mask.

        Args:
            mask (list): a list of mol mask
        """
        if mask is None:
            self.mols_mask = None
            return
        if len(mask) != len(self.mols):
            logger.error('inconsistent mask length')
            return
        if all(mask):
            self.mols_mask = None
        else:
            self.mols_mask = mask

    def get_mols_mask(self):
        """Get a list of mol mask.

        Returns:
            list: a list of mask
        """
        return self.mols_mask

    def has_mol(self, id):
        """Check if a molecule of id is defined in the moldata.

        Returns:
            bool: true is a molecule is in a moldata
        """
        return id in self.mols_index

    def get_mol(self, id):
        """Get a molecule of id.

        Args:
            id (str): id of a molecule

        Returns:
            Molecule: a molecule object
        """
        if self.has_mol(id):
            return self.mols[self.mols_index[id]]

    def get_mol_by_index(self, index):
        """Get a molecule by an index in list.

        Args:
            index (int): index in a molecule list

        Returns:
             Molecule: a molecule object
        """
        return self.mols[index]

    def get_mol_index_list(self):
        """Get a list of molecule id.

        Returns:
            list: a list of molecule id
        """
        return [m.id for m in self.mols]

    def get_smiles_list(self):
        """Get a list of molecule SMILES.

        Returns:
            list: a list of molecule SMILES
        """
        return [m.get_smiles() for m in self.mols]

    def get_safe_smiles_list(self):
        """Get a list of molecule SMILES.
        If smiles of a molecule is empty, it is replaced by '*'.

        Returns:
            list: a list of molecule SMILES
        """
        smiles_list = []
        for m in self.get_mols():
            sm = m.get_smiles()
            if sm == '':
                smiles_list.append('*')
            else:
                smiles_list.append(sm)
        return smiles_list

    def get_smarts_list(self):
        """Get a list of molecule SMARTS.

        Returns:
            list: a list of molecule SMARTS
        """
        return [m.get_smarts() for m in self.mols]

    def get_mol_list(self):
        """Get a list of rdkit mols.

        Returns:
            list: a list of rdkit mols
        """
        return [m.get_mol() for m in self.mols]

    def get_mol_image_list(self):
        """Get a list of mol images

        Returns:
            list: a list of mol images
        """
        return [Chem.Draw.MolToImage(m.get_mol()) for m in self.mols]

    def get_properties(self):
        """Get a property set.

        Returns:
            PropertySet: a property set
        """
        return self.properties

    def get_property_vector(self):
        """Get a property vectors.

        Returns:
            DataFrame: a matrix of molecules and property values
        """
        # return self.property_vector
        return self.properties.make_property_vector(self.mols)

    def get_subdata_labels(self, recursive=False):
        """Get a list of child subdata labels

        Args:
            recursive (bool): flag for getting all the offsprings

        Returns:
            list: a list of subdata labels
        """
        labels = []
        if self.anion is not None:
            labels.append('anion')
            if recursive:
                for sub_label in self.anion.get_subdata_labels(recursive=recursive):
                    labels.append('{0}:{1}'.format('anion', sub_label))
        if self.cation is not None:
            labels.append('cation')
            if recursive:
                for sub_label in self.cation.get_subdata_labels(recursive=recursive):
                    labels.append('{0}:{1}'.format('cation', sub_label))
        if self.main_ion is not None:
            labels.append('main_ion')
            if recursive:
                for sub_label in self.main_ion.get_subdata_labels(recursive=recursive):
                    labels.append('{0}:{1}'.format('main_ion', sub_label))
        if self.sub_ion is not None:
            labels.append('sub_ion')
            if recursive:
                for sub_label in self.sub_ion.get_subdata_labels(recursive=recursive):
                    labels.append('{0}:{1}'.format('sub_ion', sub_label))
        return labels

    def get_subdata(self, label):
        """Get moldata by a path of labels (e.g., anion)

        Args:
            label (str): a path of labels

        Returns:
            MolData: a moldata for the path of labels
        """

        labels = label.split(':', maxsplit=1)
        if len(labels) == 1:
            if label == '':
                return self
            elif label == 'anion':
                return self.anion
            elif label == 'cation':
                return self.cation
            elif label == 'main_ion':
                return self.main_ion
            elif label == 'sub_ion':
                return self.sub_ion
        elif len(labels) == 2:
            if labels[0] == 'anion':
                subdata = self.get_anion().get_subdata(labels[1])
                return subdata
            elif labels[0] == 'cation':
                subdata = self.get_cation().get_subdata(labels[1])
                return subdata
            elif labels[0] == 'main_ion':
                subdata = self.get_main_ion().get_subdata(labels[1])
                return subdata
            elif labels[0] == 'sub_ion':
                subdata = self.get_sub_ion().get_subdata(labels[1])
                return subdata
        return None

    def get_anion(self):
        """Get an anion part of molecules

        Returns:
            MolData: a moldata for anions
        """
        return self.anion

    def get_cation(self):
        """Get a cation part of molecules

        Returns:
            MolData: a moldata for cation
        """
        return self.cation

    def get_main_ion(self):
        """Get a main ion part of molecules

        Returns:
            MolData: a moldata for main ion
        """
        return self.main_ion

    def get_sub_ion(self):
        """Get a sub ion part of molecules

        Returns:
            MolData: a moldata for sub ion
        """
        return self.sub_ion

    def add_features(self, features, feature_mask=None, replace=False):
        """Add a set of features

        Args:
            features (FeatureSet): a set of features
            feature_mask (list): a mask of valid molecules for features
            replace (bool): flag of replacing existing entry
        """
        if replace and self.has_features(features.id):
            index = self.features_index[features.id]
            self.features_list[index] = features
            self.features_mask_list[index] = feature_mask
        else:
            index = len(self.features_list)
            self.features_list.append(features)
            self.features_mask_list.append(feature_mask)
            self.features_index[features.id] = index

    def add_merged_features(self, merged_features, merged_feature_mask=None, replace=False):
        """Add a set of merged features.

        Args:
            merged_features (FeatureSet): a set of merged features
            merged_feature_mask (list): a mask of valid molecules for merged features
            replace (bool): flag of replacing existing entry
        """
        if replace and self.has_merged_features(merged_features.id):
            index = self.merged_features_index[merged_features.id]
            self.merged_features_list[index] = merged_features
            self.merged_features_mask_list[index] = merged_feature_mask
        else:
            index = len(self.merged_features_list)
            self.merged_features_list.append(merged_features)
            self.merged_features_mask_list.append(merged_feature_mask)
            self.merged_features_index[merged_features.id] = index

    def merge_features(self, id_list, force=False):
        """Merge feature sets specified by their id and register results.

        Args:
            id_list (list): a list of ids of feature set to merge
            force (bool, optional): flag for forcefully merge features

        Returns:
            MergedFeatureSet: a set of merged feature sets
        """
        features_list = self.get_features_list(id_list)
        features_list = sorted(features_list, key=lambda x: x.id)
        feature_mask_list = self.get_feature_mask_list([f.id for f in features_list])
        # merge features
        merged_features = MergedFeatureSet(features_list)
        if self.has_merged_features(merged_features.id):
            if not force:
                logger.warning('features are already merged:id=%s', merged_features.id)
                return self.get_merged_features(merged_features.id)
        # merge feature mask
        merged_feature_mask = None
        for feature_mask in feature_mask_list:
            merged_feature_mask = update_data_mask(merged_feature_mask, feature_mask)
        self.add_merged_features(merged_features, merged_feature_mask, replace=force)
        return merged_features

    def merge_features_by_index(self, index_list, force=False):
        """Merge feature sets specified by their index and register results.

        Args:
            index_list (list): a list of indices of feature set to merge
            force (bool, optional): flag for forcefully merge features

        Returns:
            MergedFeatureSEt: a set of merged feature set
        """
        features_list = self.get_features_list_by_index(index_list)
        features_list = sorted(features_list, key=lambda x: x.id)
        feature_mask_list = self.get_feature_mask_list([f.id for f in features_list])
        # merge features
        merged_features = MergedFeatureSet(features_list)
        if self.has_merged_features(merged_features.id):
            if not force:
                logger.warning('features are already merged:id=%s', merged_features.id)
                return self.get_merged_features(merged_features.id)
        # merge feature mask
        merged_feature_mask = None
        for feature_mask in feature_mask_list:
            merged_feature_mask = update_data_mask(merged_feature_mask, feature_mask)
        self.add_merged_features(merged_features, merged_feature_mask, replace=force)
        return merged_features

    def has_features(self, id):
        """Check if a feature set is registered.

        Args:
            id (str): id of a feature set

        Returns:
            bool: true if a feature set is registered
        """
        return id in self.features_index

    def get_features(self, id):
        """Get a feature set by its id.

        Args:
            id (str): id of a feature set

        Returns:
            FeatureSet: a feature set
        """
        return self.features_list[self.features_index[id]]

    def get_features_by_index(self, index):
        """Get a feature set by its index in a list.

        Args:
            index (int): index of a feature set in a list

        Returns:
            FeatureSet: a feature set
        """
        return self.features_list[index]

    def remove_features(self, id):
        """Remove a feature set by its id.

        Args:
            id (str): id of a feature set
        """
        new_features_list = []
        new_features_mask_list = []
        new_features_index = dict()
        new_index = 0
        target_features = None
        for fs, fm in zip(self.features_list, self.features_mask_list):
            if fs.id == id:
                target_features = fs
                if self.check_merged_features_including(fs.id):
                    logger.error('feature set %s is referred in a merged feature set', fs.id)
                    return
            else:
                new_features_list.append(fs)
                new_features_mask_list.append(fm)
                new_features_index[fs.id] = new_index
                new_index += 1
        self.features_list = new_features_list
        self.features_mask_list = new_features_mask_list
        self.features_index = new_features_index
        # remove individual feature from molecules
        if target_features is not None:
            for feature in target_features.get_feature_list():
                for molecule in self.mols:
                    molecule.remove_feature(feature)
        return

    def remove_features_by_index(self, index):
        """Remove a feature set by its index in a list.

        Args:
            index (int): index of a feature set in a list
        """
        new_features_list = []
        new_features_mask_list = []
        new_features_index = dict()
        new_index = 0
        target_features = None
        for fs, fm in zip(self.features_list, self.features_mask_list):
            if new_index == index:
                target_features = fs
                index = -1
                if self.check_merged_features_including(fs.id):
                    logger.error('feature set %s is referred in a merged feature set', fs.id)
                    return
            else:
                new_features_list.append(fs)
                new_features_mask_list.append(fm)
                new_features_index[fs.id] = new_index
                new_index += 1
        self.features_list = new_features_list
        self.features_mask_list = new_features_mask_list
        self.features_index = new_features_index
        # remove individual feature from molecules
        if target_features is not None:
            for feature in target_features.get_feature_list():
                for molecule in self.mols:
                    molecule.remove_feature(feature)
        return

    def clear_features(self):
        """Remove all the feature set
        """
        # clear features in subdata
        for label in self.get_subdata_labels():
            self.get_subdata(label).clear_features()

        # clear features of this moldata
        for features in self.features_list:
            if self.check_merged_features_including(features.id):
                logger.error('feature set %s is referred in a merged feature set', features.id)
                return
        for features in self.features_list:
            for feature in features.get_feature_list():
                for molecule in self.mols:
                    molecule.remove_feature(feature)
        self.features_list = []
        self.features_mask_list = []
        self.features_index = dict()

    def get_feature_vector(self, id, readable=False):
        """Get a matrix of molecules and a feature set.

        Args:
            id (str): id of a feature set
            readable (bool, optional): flag for getting value in readable form. Default to False.

        Returns:
            DataFrame: a matrix of molecules and a feature set
        """
        fv = None
        if self.has_features(id):
            fs = self.get_features(id)
            fv = fs.make_feature_vector(self.mols, readable=readable)
        elif self.has_merged_features(id):
            fv = self.get_merged_feature_vector(id, readable=readable)
        else:
            logger.error('unknown feature:id=%s', id)
        return fv

    def get_feature_mask(self, id):
        """Get a mask of molecules for a feature set.

        Args:
            id (str): id of a feature set

        Returns:
            list: a list of masks of valid molecules
        """
        fv = None
        if self.has_features(id):
            fv = self.features_mask_list[self.features_index[id]]
        elif self.has_merged_features(id):
            fv = self.merged_features_mask_list[self.merged_features_index[id]]
        else:
            logger.error('unknown feature:id=%s', id)
        return fv

    def get_features_list(self, id_list):
        """Get a list of feature sets.

        Args:
            id_list (list): a list of ids of feature sets

        Returns:
            list: a list of feature sets
        """
        features_list = []
        for id in id_list:
            features_list.append(self.get_features(id))
        return features_list

    def get_features_list_by_index(self, index_list):
        """Get a list of feature sets.

        Args:
            index_list (list) a list of indices of feature sets

        Returns:
            list: a list of feature sets
        """
        features_list = []
        for index in index_list:
            features_list.append(self.get_features_by_index(index))
        return features_list

    def get_feature_mask_list(self, id_list):
        """Get a list of masks of valid molecules.

        Args:
            id_list (list): a list of ids of feature sets

        Returns:
            list: a list of masks of valid molecules
        """
        feature_mask_list = []
        for id in id_list:
            feature_mask_list.append(self.get_feature_mask(id))
        return feature_mask_list

    def get_feature_mask_list_by_index(self, index_list):
        """Get a list of masks of valid molecules.

        Args:
            index_list (list): a list of indices of feature sets

        Returns:
            list: a list of masks of valid molecules
        """
        feature_mask_list = []
        for index in index_list:
            feature = self.get_features_by_index(index)
            feature_mask_list.append(self.get_feature_mask(feature.id))
        return feature_mask_list

    def has_merged_features(self, id):
        """Check if a merged feature set is registered.

        Args:
            id (str): id of merged feature sets

        Returns:
            bool: true if a merged feature set is registered
        """
        return id in self.merged_features_index

    def get_merged_features(self, id):
        """Get a merged feature set by its id.

        Args:
            id (str): id of merged feature set

        Returns:
            MergedFeatureSet: a merged feature set
        """
        return self.merged_features_list[self.merged_features_index[id]]

    def get_merged_features_by_index(self, index):
        """Get a merged feature set by its index.

        Args:
            index (int): index of merged feature set in a list

        Returns:
            MergedFeatureSet: a merged feature set
        """
        return self.merged_features_list[index]

    def remove_merged_features(self, id, only_entry=False):
        """Remove a merged feature set by its id.

        Args:
            id (str): id of a feature set
            only_entry (bool): remove only entry
        """
        new_features_list = []
        new_features_mask_list = []
        new_features_index = dict()
        new_index = 0
        for fs, fm in zip(self.merged_features_list, self.merged_features_mask_list):
            if fs.id != id:
                new_features_list.append(fs)
                new_features_mask_list.append(fm)
                new_features_index[fs.id] = new_index
                new_index += 1
            else:
                if not only_entry:
                    if self.check_regression_model_including(fs.id):
                        logger.error('merged feature set %s is referred in a regression model', fs.id)
                        return
        self.merged_features_list = new_features_list
        self.merged_features_mask_list = new_features_mask_list
        self.merged_features_index = new_features_index
        return

    def remove_merged_features_by_index(self, index, only_entry=False):
        """Remove a feature set by its index in a list.

        Args:
            index (int): index of a feature set in a list
            only_entry (bool): remove only entry
        """
        new_features_list = []
        new_features_mask_list = []
        new_features_index = dict()
        new_index = 0
        for fs, fm in zip(self.merged_features_list, self.merged_features_mask_list):
            if new_index != index:
                new_features_list.append(fs)
                new_features_mask_list.append(fm)
                new_features_index[fs.id] = new_index
                new_index += 1
            else:
                index = -1
                if not only_entry:
                    if self.check_regression_model_including(fs.id):
                        logger.error('merged feature set %s is referred in a regression model', fs.id)
                        return
        self.merged_features_list = new_features_list
        self.merged_features_mask_list = new_features_mask_list
        self.merged_features_index = new_features_index
        return

    def clear_merged_features(self):
        """Remove all the merged feature set
        """
        for features in self.merged_features_list:
            if self.check_regression_model_including(features.id):
                logger.error('merged feature set %s is referred in a regression model', features.id)
                return
        self.merged_features_list = []
        self.merged_features_mask_list = []
        self.merged_features_index = dict()

    def check_merged_features_including(self, features_id):
        """Check if there is merged features including a feature set

        Args:
            features_id (str): id of a merged feature set

        Returns:
            bool: true if there is a such features
        """
        for features in self.merged_features_list:
            for fs in features.get_features_list():
                if features_id == fs.id:
                    return True
        return False

    def get_merged_feature_vector(self, id, readable=False):
        """Get a matrix of molecules and merged feature sets.

        Args:
            id (str): id of merged feature set
            readable (bool, optional): flag for getting value in readable form. Default to False.

        Returns:
            DataFrame: a matrix of molecules and merged feature set
        """
        fs = self.get_merged_features(id)
        return fs.make_feature_vector(self.mols, readable=readable)

    def get_merged_feature_vector_by_index(self, index, readable=False):
        """Get a matrix of molecules and merged feature sets.

        Args:
            index (int): index of merged feature set in a list
            readable (bool, optional): flag for getting value in readable form. Default to False.

        Returns:
            DataFrame: a matrix of molecules and merged feature set
        """
        fs = self.get_merged_features_by_index(index)
        return fs.make_feature_vector(self.mols, readable=readable)

    def get_merged_feature_mask(self, id):
        """Get a mask of valid molecules for a merged feature set.

        Args:
            id (str): id of merged feature set

        Returns:
            list: a list of masks of valid molecules
        """
        return self.merged_features_mask_list[self.merged_features_index[id]]

    def get_merged_feature_mask_by_index(self, index):
        """Get a mask of valid molecules for a merged feature set.

        Args:
            index (int): an index of merged feature set

        Returns:
            list: a list of masks of valid molecules
        """
        return self.merged_features_mask_list[index]

    def add_regression_model(self, model):
        """Register a regression model. A regression model is stored in a list classified by a target property
        and a feature set. A snapshot of a regression model is stored instead of a given regression model object.

        Args:
            model (RegressionModel): a regression model associated with a target property and a feature set
        """
        target_property = model.get_target_property()
        features = model.get_features()
        features_id = features.get_id()
        model_id = model.get_id()
        # check model in self
        if target_property in self.regression_model_index and \
                features_id in self.regression_model_index[target_property] and \
                model_id in self.regression_model_index[target_property][features_id]:
            index = self.regression_model_index[target_property][features_id][model_id]
            # make a copy of estimator
            self.regression_model_list[target_property][features_id][index] = ModelSnapShot(model)
        else:
            index = len(self.regression_model_list[target_property][features_id])
            # make a copy of estimator
            self.regression_model_list[target_property][features_id].append(ModelSnapShot(model))
            self.regression_model_index[target_property][features_id][model_id] = index

    def get_regression_model_snapshot(self, target_property, features_id, model_id):
        """Get a snapshot of a regression model.

        Args:
            target_property (str): a target property of a regression
            features_id (str): an id of feature set of a regression
            model_id (str): an id of a regression model

        Returns:
            ModelSnapShot: a snapshot of a regression model
        """
        if target_property in self.regression_model_index and \
                features_id in self.regression_model_index[target_property] and \
                model_id in self.regression_model_index[target_property][features_id]:
            index = self.regression_model_index[target_property][features_id][model_id]
            model_snapshot = self.regression_model_list[target_property][features_id][index]
            return model_snapshot
        else:
            logger.error('no model:{0} for property:{1} feature:{2}'.
                         format(model_id, target_property, features_id))
            return None

    def get_regression_model_snapshot_by_index(self, target_property, feature_id, index):
        """Get a snapshot of a regression model.

        Args:
            target_property (str): a target property of a regression
            feature_id (str): an id of feature set of a regression
            index (int): an index of a regression model in a list

        Returns:
            ModelSnapShot: a snapshot of a regression model
        """
        if target_property in self.regression_model_index and \
                feature_id in self.regression_model_index[target_property] and \
                index in self.regression_model_index[target_property][feature_id].values():
            return self.regression_model_list[target_property][feature_id][index]
        else:
            logger.error('no model for property:{0} feature:{1} index:{2}'.
                         format(target_property, feature_id, index))
            return None

    def get_regression_model(self, target_property, features_id, model_id):
        """Get a regression model.

        Args:
            target_property (str): a target property of a regression
            features_id (str): an id of feature set of a regression
            model_id (str): an id of a regression model

        Returns:
            RegressionModel: a regression model
        """
        snapshot = self.get_regression_model_snapshot(target_property, features_id, model_id)
        if snapshot is not None:
            return snapshot.get_model()
        else:
            return None

    def get_regression_model_by_index(self, target_property, feature_id, index):
        """Get a regression model.

        Args:
            target_property (str): a target property of a regression
            feature_id (str): an id of feature set of a regression
            index (int): an index of a regression model in a list

        Returns:
            RegressionModel: a regression model
        """
        snapshot = self.get_regression_model_snapshot_by_index(target_property, feature_id, index)
        if snapshot is not None:
            return snapshot.get_model()
        else:
            return None

    def remove_regression_model(self, target_property, features_id, model_id):
        """Remove a regression model.

        Args:
            target_property (str): a target property of a regression
            features_id (str): an id of feature set of a regression
            model_id (str): an id of a regression model
        """
        # check model in self
        if target_property in self.regression_model_index and \
                features_id in self.regression_model_index[target_property] and \
                model_id in self.regression_model_index[target_property][features_id]:
            index = self.regression_model_index[target_property][features_id][model_id]
            # remove a copy of estimator
            del self.regression_model_list[target_property][features_id][index]
            del self.regression_model_index[target_property][features_id][model_id]
            if len(self.regression_model_list[target_property][features_id]) == 0:
                del self.regression_model_list[target_property][features_id]
                del self.regression_model_index[target_property][features_id]
                if len(self.regression_model_list[target_property]) == 0:
                    del self.regression_model_list[target_property]
                    del self.regression_model_index[target_property]
            else:
                # update index of other models
                for models_id0, index0 in self.regression_model_index[target_property][features_id].items():
                    if index0 > index:
                        self.regression_model_index[target_property][features_id][models_id0] -= 1

    def remove_regression_model_by_index(self, target_property, features_id, index):
        """Remove a regression model.

        Args:
            target_property (str): a target property of a regression
            features_id (str): an id of feature set of a regression
            index (int): an index of a regression model in a list
        """
        # check model in self
        if target_property in self.regression_model_index and \
                features_id in self.regression_model_index[target_property] and \
                index in self.regression_model_index[target_property][features_id].values():
            model_id = self.regression_model_list[target_property][features_id][index].get_id()
            # remove a copy of estimator
            del self.regression_model_list[target_property][features_id][index]
            del self.regression_model_index[target_property][features_id][model_id]
            if len(self.regression_model_list[target_property][features_id]) == 0:
                del self.regression_model_list[target_property][features_id]
                del self.regression_model_index[target_property][features_id]
                if len(self.regression_model_list[target_property]) == 0:
                    del self.regression_model_list[target_property]
                    del self.regression_model_index[target_property]
            else:
                # update index of other models
                for models_id0, index0 in self.regression_model_index[target_property][features_id].items():
                    if index0 > index:
                        self.regression_model_index[target_property][features_id][models_id0] -= 1

    def clear_regression_model(self, target_property=None, features_id=None):
        """Remove all the regression model.

        Args:
            target_property (str, optional): a target property of a regression. Defaults to None.
            features_id (str, optional): an id of feature set of a regression. Defaults to None.
        """
        # check argument
        target_properties = []
        if target_property is None:
            target_properties.extend(self.regression_model_index.keys())
        else:
            target_properties.append(target_property)
        for target_property0 in target_properties:
            features_ids = []
            if features_id is None:
                features_ids.extend(self.regression_model_index[target_property0].keys())
            else:
                features_ids.append(features_id)
            for features_id0 in features_ids:
                # check model in self
                if target_property0 in self.regression_model_index and \
                        features_id0 in self.regression_model_index[target_property0]:
                    # clear a copy of estimator
                    del self.regression_model_list[target_property0][features_id0]
                    del self.regression_model_index[target_property0][features_id0]
                    if len(self.regression_model_list[target_property0]) == 0:
                        del self.regression_model_list[target_property0]
                        del self.regression_model_index[target_property0]

    def check_regression_model_including(self, features_id):
        """Check if there is a regression model using a merged feature

        Args:
            features_id (str): id of a merged feature set

        Returns:
            bool: true if there is a regression model
        """
        for target_property in self.regression_model_index:
            if features_id in self.regression_model_index[target_property]:
                return True
        return False

    def get_regression_model_summary(self, target_property, models=None, features=None, legends=None):
        """Get summary of regression models for a target property

        Args:
            target_property (str): name of target property of regression
            models (list): a list of models as a filter
            features (list): a list of feature sets as a filter
            legends (dict, optional): a mapping of feature set id and a name. Defaults to None

        Returns:
            DataFrame: a table of model fitting results
        """
        # make a list of features
        if target_property not in self.regression_model_index:
            logger.error('no model for property \'%s\'', target_property)
            return None

        # make a features list of target property
        features_list = []
        features_name = dict()
        features_length = Counter()
        target_features = set()
        for features_id in self.regression_model_index[target_property]:
            if self.has_merged_features(features_id):
                m_features = self.get_merged_features(features_id)
                # check features
                if features is None:
                    target_features.add(features_id)
                else:
                    for f in m_features.get_features_list():
                        if f.id in features:
                            target_features.add(features_id)
                            break
                if features_id not in target_features:
                    continue
                for f in m_features.get_features_list():
                    id = ':' + f.id
                    name = f.id if legends is None else (legends[f.id] if f.id in legends else f.id)
                    if id not in features_name:
                        features_list.append(('', f.id))
                        features_name[id] = name
                        features_length[id] = f.get_vector_size()
        features_list = sorted(features_list)

        # get model results
        result_value_map = defaultdict(list)
        for features_id in self.regression_model_index[target_property]:
            if features_id not in target_features:
                continue
            features_slice = dict()
            if self.has_merged_features(features_id):
                m_features = self.get_merged_features(features_id)
                index = 0
                for f in m_features.get_features_list():
                    id = ':' + f.id
                    features_slice[id] = slice(index, index + f.get_vector_size())
                    index += f.get_vector_size()
            # get models
            for model_snapshot in self.regression_model_list[target_property][features_id]:
                model = model_snapshot.get_model()
                # check model
                good_model = True
                if models is not None:
                    good_model = False
                    for m_cls in models:
                        if isinstance(model, m_cls):
                            good_model = True
                            break
                if not good_model:
                    continue
                # make a dict for making results dataframe
                result_value_map['model'].append(model)
                result_value_map['model_id'].append(model.get_id())
                result_value_map['score'].append(model.get_score())
                result_value_map['cv_score'].append(model.get_cv_score_mean())
                result_value_map['cv_score(std)'].append(model.get_cv_score_std())
                result_value_map['rmse'].append(model.get_rmse())
                result_value_map['size'].append(model.get_vector_size())
                selection_mask = model.get_selection_mask() if model.is_feature_selected() else None
                for label, f_id in features_list:
                    feature_id = label+':'+f_id
                    map_id = '{0} ({1})'.format(features_name[feature_id], features_length[feature_id])
                    if feature_id in features_slice:
                        # get selected size
                        f_slice = features_slice[feature_id]
                        if selection_mask is None:
                            selection_size = f_slice.stop - f_slice.start
                        else:
                            selection_size = len([s for s in selection_mask[f_slice] if s])
                        result_value_map[map_id].append(selection_size)
                    else:
                        result_value_map[map_id].append('')

        # make a dataframe
        if len(result_value_map) > 0:
            df = pd.DataFrame(result_value_map)
            df = df.sort_values(['cv_score', 'cv_score(std)', 'score'], ascending=[False, True, False])
            # renumber index of dataframe by sorted result
            for index, df_index in enumerate(df.index.values):
                df.index.values[index] = index
            return df
        else:
            return None

    def has_feature_estimate_by_models(self, models):
        """Check if a feature estimate exists

        Args:
            models (list): a list of regression models

        Returns:
            bool: true if exists
        """
        # check moldata of models
        moldata = None
        for model in models:
            if moldata is not None and moldata != model.get_moldata():
                return False
            elif moldata is None:
                moldata = model.get_moldata()
        target_property_id = FeatureEvaluator.get_target_property_id_string(models)
        features_id = FeatureEvaluator.get_features_id_string(models)
        models_id = FeatureEvaluator.get_models_id_string(models)
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                models_id in self.feature_estimate_index[target_property_id][features_id]:
            return True
        else:
            return False

    def add_feature_estimate(self, feature_estimate):
        """Register a result of feature estimation. The feature estimate is stored in a list
        classified by a target property and a feature set.

        Args:
            feature_estimate (FeatureEstimationResult): a result of feature estimate
        """
        # check label of feature estimate
        moldata = feature_estimate.get_evaluator().get_moldata()
        label = feature_estimate.get_label()
        if self != moldata.get_subdata(label):
            logger.error('feature estimate for %s cannot add moldata for %s', label, self.get_mol_label())
            return
        target_property_id = feature_estimate.get_target_property_id()
        features_id = feature_estimate.get_features_id()
        models_id = feature_estimate.get_models_id()
        # check feature estimate in subdata
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                models_id in self.feature_estimate_index[target_property_id][features_id]:
            index = self.feature_estimate_index[target_property_id][features_id][models_id]
            self.feature_estimate_list[target_property_id][features_id][index] = feature_estimate
        else:
            index = len(self.feature_estimate_list[target_property_id][features_id])
            self.feature_estimate_list[target_property_id][features_id].append(feature_estimate)
            self.feature_estimate_index[target_property_id][features_id][models_id] = index

    def get_feature_estimate(self, target_property_id, features_id, models_id):
        """Get a feature estimate.

        Args:
            target_property_id (str): id of target properties
            features_id (str): id of merged feature sets
            models_id (str): id of regression models

        Returns:
            FeatureEstimationResult: a result of feature estimation
        """
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                models_id in self.feature_estimate_index[target_property_id][features_id]:
            index = self.feature_estimate_index[target_property_id][features_id][models_id]
            feature_estimate = self.feature_estimate_list[target_property_id][features_id][index]
            return feature_estimate
        else:
            logger.error('no feature estimate for property:{0} feature:{1} model:{2}'.format(target_property_id,
                                                                                             features_id, models_id))
            return None

    def get_feature_estimate_by_index(self, target_property_id, features_id, index):
        """Get a snapshot of a regression model.

        Args:
            target_property_id (str): id of target properties
            features_id (str): id of merged feature sets
            index (int): an index of a feature estimate in a list

        Returns:
            FeatureEstimationResult: a result of feature estimation
        """
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                index in self.feature_estimate_index[target_property_id][features_id].values():
            feature_estimate = self.feature_estimate_list[target_property_id][features_id][index]
            return feature_estimate
        else:
            logger.error('no feature estimate for property:{0} feature:{1}'.
                         format(target_property_id, features_id))
            return None

    def get_feature_estimate_by_models(self, models):
        """Get a feature estimate.

        Args:
            models (list): a list of regression models

        Returns:
            FeatureEstimationResult: a result of feature estimation
        """
        moldata = None
        for model in models:
            if moldata is not None and moldata != model.get_moldata():
                return None
            elif moldata is None:
                moldata = model.get_moldata()
        target_property_id = FeatureEvaluator.get_target_property_id_string(models)
        features_id = FeatureEvaluator.get_features_id_string(models)
        models_id = FeatureEvaluator.get_models_id_string(models)
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                models_id in self.feature_estimate_index[target_property_id][features_id]:
            index = self.feature_estimate_index[target_property_id][features_id][models_id]
            feature_estimate = self.feature_estimate_list[target_property_id][features_id][index]
            return feature_estimate
        else:
            return None

    def remove_feature_estimate(self, target_property_id, features_id, models_id):
        """Remove a feature estimate

        Args:
            target_property_id (str): id of target properties
            features_id (str): id of merged feature sets
            models_id (str): id of regression models
        """
        # check feature estimate in self
        labels = []
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                models_id in self.feature_estimate_index[target_property_id][features_id]:
            index = self.feature_estimate_index[target_property_id][features_id][models_id]
            feature_estimate = self.feature_estimate_list[target_property_id][features_id][index]
            if feature_estimate.get_label() != '':
                return
            labels = feature_estimate.get_evaluator().get_labels()
            del self.feature_estimate_list[target_property_id][features_id][index]
            del self.feature_estimate_index[target_property_id][features_id][models_id]
            if len(self.feature_estimate_list[target_property_id][features_id]) == 0:
                del self.feature_estimate_list[target_property_id][features_id]
                del self.feature_estimate_index[target_property_id][features_id]
                if len(self.feature_estimate_list[target_property_id]) == 0:
                    del self.feature_estimate_list[target_property_id]
                    del self.feature_estimate_index[target_property_id]
            else:
                # update index of other models
                for models_id0, index0 in self.feature_estimate_index[target_property_id][features_id].items():
                    if index0 > index:
                        self.feature_estimate_index[target_property_id][features_id][models_id0] -= 1
        for label in labels:
            if label == '':
                continue
            subdata = self.get_subdata(label)
            if target_property_id in subdata.feature_estimate_index and \
                    features_id in subdata.feature_estimate_index[target_property_id] and \
                    models_id in subdata.feature_estimate_index[target_property_id][features_id]:
                index = subdata.feature_estimate_index[target_property_id][features_id][models_id]
                del subdata.feature_estimate_list[target_property_id][features_id][index]
                del subdata.feature_estimate_index[target_property_id][features_id][models_id]
                if len(subdata.feature_estimate_list[target_property_id][features_id]) == 0:
                    del subdata.feature_estimate_list[target_property_id][features_id]
                    del subdata.feature_estimate_index[target_property_id][features_id]
                    if len(subdata.feature_estimate_list[target_property_id]) == 0:
                        del subdata.feature_estimate_list[target_property_id]
                        del subdata.feature_estimate_index[target_property_id]
                else:
                    # update index of other models
                    for models_id0, index0 in subdata.feature_estimate_index[target_property_id][features_id].items():
                        if index0 > index:
                            subdata.feature_estimate_index[target_property_id][features_id][models_id0] -= 1

    def remove_feature_estimate_by_index(self, target_property_id, features_id, index):
        """Remove a feature estimate

        Args:
            target_property_id (str): id of target properties
            features_id (str): id of merged feature sets
            index (int): an index of a feature estimate in a list
        """
        # check feature estimate in self
        labels = []
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                index in self.feature_estimate_index[target_property_id][features_id].values():
            models_id = self.feature_estimate_list[target_property_id][features_id][index].get_models_id()
            feature_estimate = self.feature_estimate_list[target_property_id][features_id][index]
            if feature_estimate.get_label() != '':
                return
            labels = feature_estimate.get_evaluator().get_labels()
            # remove a copy of estimator
            del self.feature_estimate_list[target_property_id][features_id][index]
            del self.feature_estimate_index[target_property_id][features_id][models_id]
            if len(self.feature_estimate_list[target_property_id][features_id]) == 0:
                del self.feature_estimate_list[target_property_id][features_id]
                del self.feature_estimate_index[target_property_id][features_id]
                if len(self.feature_estimate_list[target_property_id]) == 0:
                    del self.feature_estimate_list[target_property_id]
                    del self.feature_estimate_index[target_property_id]
            else:
                # update index of other models
                for models_id0, index0 in self.feature_estimate_index[target_property_id][features_id].items():
                    if index0 > index:
                        self.feature_estimate_index[target_property_id][features_id][models_id0] -= 1
        for label in labels:
            if label == '':
                continue
            subdata = self.get_subdata(label)
            if target_property_id in subdata.feature_estimate_index and \
                    features_id in subdata.feature_estimate_index[target_property_id] and \
                    models_id in subdata.feature_estimate_index[target_property_id][features_id]:
                index = subdata.feature_estimate_index[target_property_id][features_id][models_id]
                del subdata.feature_estimate_list[target_property_id][features_id][index]
                del subdata.feature_estimate_index[target_property_id][features_id][models_id]
                if len(subdata.feature_estimate_list[target_property_id][features_id]) == 0:
                    del subdata.feature_estimate_list[target_property_id][features_id]
                    del subdata.feature_estimate_index[target_property_id][features_id]
                    if len(subdata.feature_estimate_list[target_property_id]) == 0:
                        del subdata.feature_estimate_list[target_property_id]
                        del subdata.feature_estimate_index[target_property_id]
                else:
                    # update index of other models
                    for models_id0, index0 in subdata.feature_estimate_index[target_property_id][features_id].items():
                        if index0 > index:
                            subdata.feature_estimate_index[target_property_id][features_id][models_id0] -= 1

    def remove_feature_estimate_by_models(self, models):
        """Remove a feature estimate

        Args:
            models (list): a list of regression models
        """
        for model in models:
            if self != model.get_moldata():
                return
        # check label component
        labels = []
        target_property_id = FeatureEvaluator.get_target_property_id_string(models)
        features_id = FeatureEvaluator.get_features_id_string(models)
        models_id = FeatureEvaluator.get_models_id_string(models)
        if target_property_id in self.feature_estimate_index and \
                features_id in self.feature_estimate_index[target_property_id] and \
                models_id in self.feature_estimate_index[target_property_id][features_id]:
            index = self.feature_estimate_index[target_property_id][features_id][models_id]
            feature_estimate = self.feature_estimate_list[target_property_id][features_id][index]
            labels = feature_estimate.get_evaluator().get_labels()
            del self.feature_estimate_list[target_property_id][features_id][index]
            del self.feature_estimate_index[target_property_id][features_id][models_id]
            if len(self.feature_estimate_list[target_property_id][features_id]) == 0:
                del self.feature_estimate_list[target_property_id][features_id]
                del self.feature_estimate_index[target_property_id][features_id]
                if len(self.feature_estimate_list[target_property_id]) == 0:
                    del self.feature_estimate_list[target_property_id]
                    del self.feature_estimate_index[target_property_id]
            else:
                # update index of other models
                for models_id0, index0 in self.feature_estimate_index[target_property_id][features_id].items():
                    if index0 > index:
                        self.feature_estimate_index[target_property_id][features_id][models_id0] -= 1
        # remove for subdata
        for label in labels:
            if label == '':
                continue
            subdata = self.get_subdata(label)
            if target_property_id in subdata.feature_estimate_index and \
                    features_id in subdata.feature_estimate_index[target_property_id] and \
                    models_id in subdata.feature_estimate_index[target_property_id][features_id]:
                index = subdata.feature_estimate_index[target_property_id][features_id][models_id]
                del subdata.feature_estimate_list[target_property_id][features_id][index]
                del subdata.feature_estimate_index[target_property_id][features_id][models_id]
                if len(subdata.feature_estimate_list[target_property_id][features_id]) == 0:
                    del subdata.feature_estimate_list[target_property_id][features_id]
                    del subdata.feature_estimate_index[target_property_id][features_id]
                    if len(subdata.feature_estimate_list[target_property_id]) == 0:
                        del subdata.feature_estimate_list[target_property_id]
                        del subdata.feature_estimate_index[target_property_id]
                else:
                    # update index of other models
                    for models_id0, index0 in subdata.feature_estimate_index[target_property_id][features_id].items():
                        if index0 > index:
                            subdata.feature_estimate_index[target_property_id][features_id][models_id0] -= 1

    def clear_feature_estimate(self, target_property_id=None, features_id=None):
        """Remove all the feature estimates

        Args:
            target_property_id (str, optional): id of target properties. Defaults to None.
            features_id (str, optional): id of merged feature sets. Defaults to None.
        """
        # check argument
        target_properties = []
        if target_property_id is None:
            target_properties.extend(self.feature_estimate_index.keys())
        else:
            target_properties.append(target_property_id)
        for target_property_id0 in target_properties:
            features_ids = []
            if features_id is None:
                features_ids.extend(self.feature_estimate_index[target_property_id0].keys())
            else:
                features_ids.append(features_id)
            for features_id0 in features_ids:
                # check feature estimate in self
                if target_property_id0 in self.feature_estimate_index and \
                        features_id0 in self.feature_estimate_index[target_property_id0]:
                    # clear a copy of estimator
                    del self.feature_estimate_list[target_property_id0][features_id0]
                    del self.feature_estimate_index[target_property_id0][features_id0]
                    if len(self.feature_estimate_list[target_property_id0]) == 0:
                        del self.feature_estimate_list[target_property_id0]
                        del self.feature_estimate_index[target_property_id0]

    def has_feature_estimate_candidates(self, models, design_param, fix_condition=None):
        """Check if feature estimation is already done

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.

        Returns:
              bool: True if feature estimation is already done
        """
        feature_estimate = self.get_feature_estimate_by_models(models)
        if feature_estimate is None:
            return False
        if fix_condition is None:
            fix_condition = ComponentFixCondition(feature_estimate)
        return feature_estimate.has_candidates(design_param, fix_condition)

    def get_feature_estimate_candidate(self, models, design_param, id, fix_condition=None):
        """Get candidate of feature estimation

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            id (str): id of a candidate
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.

        Returns:
            FeatureEstimationResult.Candidate: a candidate of feature estimation
        """
        feature_estimate = self.get_feature_estimate_by_models(models)
        if feature_estimate is None:
            return None
        if fix_condition is None:
            fix_condition = ComponentFixCondition(feature_estimate)
        return feature_estimate.get_candidate(design_param, fix_condition, id)

    def get_feature_estimate_candidate_by_index(self, models, design_param, index, fix_condition=None):
        """Get candidate of feature estimation

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            index (int): index of a candidate
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.

        Returns:
            FeatureEstimationResult.Candidate: a candidate of feature estimation
        """
        feature_estimate = self.get_feature_estimate_by_models(models)
        if feature_estimate is None:
            return None
        if fix_condition is None:
            fix_condition = ComponentFixCondition(feature_estimate)
        return feature_estimate.get_candidate_by_index(design_param, fix_condition, index)

    def remove_feature_estimate_candidate(self, models, design_param, id, fix_condition=None):
        """Remove candidate of feature estimation

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            id (str): id of a candidate
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
        """
        feature_estimate = self.get_feature_estimate_by_models(models)
        if feature_estimate is not None:
            if fix_condition is None:
                fix_condition = ComponentFixCondition(feature_estimate)
            feature_estimate.remove_candidate(design_param, fix_condition, id)
            # remove candidate from child feature estimates
            for label in feature_estimate.get_child_labels():
                if label not in fix_condition.get_labels():
                    child_feature_estimate = feature_estimate.get_child_feature_estimate(label)
                    child_feature_estimate.remove_candidate(design_param, fix_condition, id)

    def remove_feature_estimate_candidate_by_index(self, models, design_param, index, fix_condition=None):
        """Remove candidate of feature estimation by index

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            index (int): index of a candidate
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
        """
        feature_estimate = self.get_feature_estimate_by_models(models)
        if feature_estimate is not None:
            if fix_condition is None:
                fix_condition = ComponentFixCondition(feature_estimate)
            feature_estimate.remove_candidate_by_index(design_param, fix_condition, index)
            # remove candidate from child feature estimates
            for label in feature_estimate.get_child_labels():
                if label not in fix_condition.get_labels():
                    child_feature_estimate = feature_estimate.get_child_feature_estimate(label)
                    child_feature_estimate.remove_candidate_by_index(design_param, fix_condition, index)

    def clear_feature_estimate_candidates(self, models, design_param, fix_condition=None):
        """Clear all the candidate of feature estimation

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
        """
        feature_estimate = self.get_feature_estimate_by_models(models)
        if feature_estimate is not None:
            if fix_condition is None:
                fix_condition = ComponentFixCondition(feature_estimate)
            feature_estimate.clear_candidates(design_param, fix_condition)
            # remove candidate from child feature estimates
            for label in feature_estimate.get_child_labels():
                if label not in fix_condition.get_labels():
                    child_feature_estimate = feature_estimate.get_child_feature_estimate(label)
                    child_feature_estimate.clear_candidates(design_param, fix_condition)

    def clear_feature_estimate_molecules(self, models, design_param, id, fix_condition=None):
        """Clear molecules generated from candidate of feature estimation

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            id (str): id of a candidate
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
        """
        candidate = self.get_feature_estimate_candidate(models, design_param, id, fix_condition)
        if candidate is not None:
            candidate.set_molecules(None)
            candidate.set_feature_extracted(False)

    def clear_feature_estimate_molecules_by_index(self, models, design_param, index, fix_condition=None):
        """Clear molecules generated from candidate of feature estimation

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            index (int): index of a candidate
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
        """
        candidate = self.get_feature_estimate_candidate_by_index(models, design_param, index, fix_condition)
        if candidate is not None:
            candidate.set_molecules(None)
            candidate.set_feature_extracted(False)

    def get_generated_molecules(self, models, design_param, fix_condition=None):
        """get generated molecules for models and a design parameter.

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.

        Returns:
            list: a list of lists of molecules
        """
        # check models
        for model in models:
            if isinstance(model.get_features(), MergedFeatureSet):
                if self != model.get_moldata():
                    logger.error('model has moldata not consistent with this moldata')
                    return []

        # check given design parameters
        if len(models) != len(design_param.get_target_values()):
            logger.error('invalid design parameter')
            return []
        target_properties = set()
        for model in models:
            if model.get_target_property() not in design_param.get_target_values():
                logger.error('target property %s in model is no in design parameter')
                return []
            if model.get_target_property() in target_properties:
                logger.error('duplicated target property in models')
                return []
            target_properties.add(model.get_target_property())

        # get estimated feature vectors
        feature_estimate = self.get_feature_estimate_by_models(models)
        if fix_condition is None:
            fix_condition = ComponentFixCondition(feature_estimate)
        if feature_estimate is None:
            target_property_id = FeatureEvaluator.get_target_property_id_string(models)
            models_id = FeatureEvaluator.get_models_id_string(models)
            logger.error('no feature estimate for target property \'%s\' by model %s.',
                         target_property_id, models_id)
            return []

        # check feasibility of generation
        label = feature_estimate.get_label()
        evaluator = feature_estimate.get_evaluator()
        if label == '' and len([lb for lb in evaluator.get_labels() if lb != '']) > 0:
            logger.error('feature estimate is infeasible for molecule generation')
            return []

        molecules = []
        if feature_estimate.has_candidates(design_param, fix_condition):
            candidates = feature_estimate.get_candidates(design_param, fix_condition)
            for candidate in candidates:
                candidate_molecules = candidate.get_molecules()
                if candidate_molecules is not None:
                    molecules.append(candidate.get_molecules())
        return molecules

    def ion_separation(self):
        """Separate molecule into anion/cation
        """
        anion_data = []
        cation_data = []
        anion_mask = []
        cation_mask = []
        has_ion = False
        for molecule in self.get_mols():
            if molecule.ion_separation():
                has_ion = True
            anion_data.append(molecule.get_anion())
            anion_mask.append(molecule.get_anion() is not None)
            cation_data.append(molecule.get_cation())
            cation_mask.append(molecule.get_cation() is not None)
        if not has_ion:
            logger.warning('no ion data')
            return
        if all(anion_mask):
            anion_mask = None
        self.anion = MolData(anion_data, mol_type=MolType.SIMPLE, mol_label='anion')
        self.anion.mols_mask = anion_mask
        if all(cation_mask):
            cation_mask = None
        self.cation = MolData(cation_data, mol_type=MolType.SIMPLE, mol_label='cation')
        self.cation.mols_mask = cation_mask
        # remove main/sub separation
        self.main_ion = None
        self.sub_ion = None

    def ion_separation_by_matching(self, sub_structure):
        """Separate molecule into main/sub ions based on the matching to given sub_structure.

        Args:
            sub_structure (str): SMILES of sub_structure
        """
        fragment = ChemFragment(AtomGraph(mol=Chem.MolFromSmiles(sub_structure)))
        main_data = []
        sub_data = []
        main_mask = []
        sub_mask = []
        has_ion = False
        for molecule in self.get_mols():
            if molecule.ion_separation_by_matching(fragment):
                has_ion = True
            main_data.append(molecule.get_main_ion())
            main_mask.append(molecule.get_main_ion() is not None)
            sub_data.append(molecule.get_sub_ion())
            sub_mask.append(molecule.get_sub_ion() is not None)
        if not has_ion:
            logger.warning('no ion data')
            return
        # make moldata
        if all(main_mask):
            main_mask = None
        self.main_ion = MolData(main_data, mol_type=MolType.SIMPLE, mol_label='main_ion')
        self.main_ion.mols_mask = main_mask
        if all(sub_mask):
            sub_mask = None
        self.sub_ion = MolData(sub_data, mol_type=MolType.SIMPLE, mol_label='sub_ion')
        self.sub_ion.mols_mask = sub_mask
        # remove anion/cation separation
        self.anion = None
        self.cation = None

    def ion_separation_by_weight(self):
        """Separate molecule into main/sub ions based on the weight
        """
        main_data = []
        sub_data = []
        main_mask = []
        sub_mask = []
        has_ion = False
        for molecule in self.get_mols():
            if molecule.ion_separation_by_weight():
                has_ion = True
            main_data.append(molecule.get_main_ion())
            main_mask.append(molecule.get_main_ion() is not None)
            sub_data.append(molecule.get_sub_ion())
            sub_mask.append(molecule.get_sub_ion() is not None)
        if not has_ion:
            logger.warning('no ion data')
            return
        # make moldata
        if all(main_mask):
            main_mask = None
        self.main_ion = MolData(main_data, mol_type=MolType.SIMPLE, mol_label='main_ion')
        self.main_ion.mols_mask = main_mask
        if all(sub_mask):
            sub_mask = None
        self.sub_ion = MolData(sub_data, mol_type=MolType.SIMPLE, mol_label='sub_ion')
        self.sub_ion.mols_mask = sub_mask
        # remove anion/cation separation
        self.anion = None
        self.cation = None

    def extract_features(self, feature_extractor, force=False, recursive=True):
        """Extract features, and register the result to moldata.

        Args:
            feature_extractor (FeatureExtractor): feature extractor object
            force (bool, optional): flag of forcefully extract feature. Default to False.
            recursive(bool, optional): flag of recursive application to subdata

        Returns:
            FeatureSet: a set of extracted features
        """
        if self.has_features(feature_extractor.get_id()):
            if not force:
                logger.warning('features are already extracted:id=%s', feature_extractor.get_id())
                return self.get_features(feature_extractor.get_id())
        features, feature_mask = feature_extractor.extract()
        if features is not None:
            logger.info('{0}: extracted {1} unique features [{2}]'.
                  format(feature_extractor.get_id(), features.get_size(), self.mol_label))
            self.add_features(features, feature_mask, replace=force)

        # apply the same feature extractors to subdata
        if recursive:
            for label in self.get_subdata_labels():
                subdata = self.get_subdata(label)
                extractor = copy.copy(feature_extractor)
                extractor.moldata = subdata
                if features is None:
                    features = subdata.extract_features(extractor)
                else:
                    subdata.extract_features(extractor)
        return features

    def extract_features_with_class_object(self, feature_extractor_cls, args):
        """Extract features by extractor created from class object, and register the result to moldata.

        Args:
            feature_extractor_cls (Class): class of feature extractor
            args (dict): arguments to instance creation

        Returns:
             FeatureSet: a set of extracted features
        """
        feature_extractor = feature_extractor_cls(self, **args)
        return self.extract_features(feature_extractor)

    def extract_same_features(self, mols):
        """Extract same features registered in moldata from other molecules

        Args:
            mols (list): a list of molecules
        """
        for features in self.features_list:
            features.extract_features(mols)

    def fit_regression_model(self, model, n_splits=3, shuffle=True, verbose=True):
        """Fit a regression model to moldata by cross validation.

        Args:
            model (RegressionModel): a regression model
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            RegressionModel: a fitted regression model
        """
        # check consistency of moldata of model
        if self != model.get_moldata():
            logger.error('model has moldata not consistent with this moldata')
            return None
        model.cross_validation(n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        model.register_model()
        return model

    def optimize_regression_model(self, model, param_grid=None, n_splits=3, shuffle=True, verbose=True):
        """Optimize hyperparameters of a regression model by fitting to moldata, and register
        the optimized model to moldata.

        Args:
            model (RegressionModel): a regression model
            param_grid (dict, optional): a map of a hyperparameter to the values for grid search. Defaults to None.
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            RegressionModel: an optimized regression model
        """
        if self != model.get_moldata():
            logger.error('model has moldata not consistent with this moldata')
            return None
        model.param_optimization(param_grid=param_grid, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        model.register_model()
        return model

    def select_features(self, model, threshold=None, n_splits=3, shuffle=True, verbose=True):
        """Select appropriate features for a regression model, and register the model to moldata.

        Args:
            model (RegressionModel): a regression model
            threshold (str, float, optional): threshold of coefficient for eliminating meaningless features.
                Defaults to None.
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            RegressionModel: a regression model with selected features
        """
        if self != model.get_moldata():
            logger.error('model has moldata not consistent with this moldata')
            return None
        model.feature_selection(threshold=threshold, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        model.register_model()
        return model

    def optimize_and_select_features(self, model, param_grid=None, threshold=None,
                                     n_splits=3, shuffle=True, verbose=True):
        """Select appropriate features for a regression model after hyperparameters of the model
        is optimized, and register the model to moldata.

        Args:
            model (RegressionModel): a regression model
            param_grid (dict, optional): a map of a hyperparameter to the values for grid search. Defaults to None.
            threshold (str, float, optional): threshold of coefficient for eliminating meaningless features.
                Defaults to None.
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            RegressionModel: a regression model with selected features
        """
        if self != model.get_moldata():
            logger.error('model has moldata not consistent with this moldata')
            return None
        model.param_optimization(param_grid=param_grid, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        model.feature_selection(threshold=threshold, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        model.register_model()
        return model

    def make_design_parameter(self, target_values, 
                              max_atom=0, max_ring=0, extend_solution=False,
                              sigma_ratio=2.0, count_tolerance=0, count_min_ratio=None, count_max_ratio=None,
                              prediction_error=1.0):
        """Make design parameters for the inverse design (feature estimation, and molecule generation).

        Args:
            target_values (dict): mapping a property name and a target value
            max_atom (int, optional): upper bound on the heavy atom number. Defaults to 0.
            max_ring (int, optional): upper bound on the ring number. Defaults to 0.
            extend_solution (bool, optional): flag to extend the range of feature vector values. Defaults to False
            sigma_ratio (float, optional): std multiplier for search range. Defaults to 2.0
            count_tolerance (int, optional): tolerance of counting feature for search range. Default to 0
            count_min_ratio (float, optional): min value multiplier for search range. Defaults to None
            count_max_ratio (float, optional): max value multiplier for search range. Defaults to None
            prediction_error (float, optional): acceptable ratio of prediction error. Defaults to 1.0

        Returns:
            DesignParameter: a design parameter object
        """
        # check target property
        for target_property in target_values.keys():
            if not self.get_properties().has_property(target_property):
                logger.error('moldata has no such property: %s', target_property)
                return None

        # update target values
        new_target_values = dict()
        for prop, target_value in target_values.items():
            if isinstance(target_value, tuple):
                if len(target_value) != 2:
                    raise ValueError('invalid target value:{0}'.format(target_value))
                elif target_value[0] > target_value[1]:
                    raise ValueError('invalid target value:{0}'.format(target_value))
                new_target_value = target_value
            elif isinstance(target_value, int) or isinstance(target_value, float):
                new_target_value = (target_value, target_value)
            else:
                raise ValueError('invalid target value:{0}'.format(target_value))
            new_target_values[prop] = new_target_value

        if prediction_error == 0.0:
            raise ValueError('prediction_error should not be zero')

        range_params = {
            'max_atom': max_atom,
            'max_ring': max_ring,
            'extend_solution': extend_solution,
            'sigma_ratio': sigma_ratio,
            'count_tolerance': count_tolerance,
            'count_min_ratio': count_min_ratio,
            'count_max_ratio': count_max_ratio,
            'prediction_error': prediction_error,
        }

        design_param = DesignParam(new_target_values, range_params)
        return design_param

    def estimate_feature(self, models, design_param, fix_condition=None, duplication_check='',
                         num_candidate=1, num_particle=1000, max_iteration=1000, renew=False, verbose=False):
        """Estimate features for target value of the property of a model. Estimated feature vector is
        stored in a moldata.

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
            duplication_check (str, optional): label of component for duplication check. Defaults to ''.
            num_candidate (int, optional): number of candidate feature vectors. Defaults to 1.
            num_particle (int, optional): number of particles for optimization. Defaults to 1000.
            max_iteration (int, optional): maximum number of iteration for optimization. Defaults to 1000.
            renew (bool): overwrite existing feature estimates
            verbose (bool, optional): flag of verbose message. Defaults to False.

        Returns:
            FeatureEstimationResult: result of feature estimate
        """
        # check models
        for model in models:
            if self != model.get_moldata():
                logger.error('moldata of a model is in consistent with this moldata')
                return None

        # check given design parameters
        if len(models) != len(design_param.get_target_values()):
            logger.error('invalid design parameter')
            return None
        target_properties = set()
        for model in models:
            if model.get_target_property() not in design_param.get_target_values():
                logger.error('target property %s in model is no in design parameter')
                return None
            if model.get_target_property() in target_properties:
                logger.error('duplicated target property in models')
                return None
            target_properties.add(model.get_target_property())

        # get estimated feature vectors
        if self.has_feature_estimate_by_models(models):
            old_feature_estimate = self.get_feature_estimate_by_models(models)
            if fix_condition is None:
                fix_condition = ComponentFixCondition(old_feature_estimate)
            old_estimates = []
            # check existing candidates
            if old_feature_estimate.has_candidates(design_param, fix_condition):
                if renew:
                    # clear candidates for design_param and fix_condition
                    self.clear_feature_estimate_candidates(models, design_param, fix_condition)
                # get existing candidates as old ones
                candidates = old_feature_estimate.get_candidates(design_param, fix_condition)
                for candidate in candidates:
                    old_estimates.append(candidate.get_whole_feature_vector())
            # estimate new feature vectors
            feature_evaluator = old_feature_estimate.get_evaluator()
            feature_estimator = FeatureEstimator(feature_evaluator)
            feature_estimate = feature_estimator.estimate(design_param, fix_condition, duplication_check,
                                                          num_candidate=num_candidate,
                                                          old_estimates=old_estimates,
                                                          num_particle=num_particle,
                                                          max_iteration=max_iteration,
                                                          verbose=verbose)

            # add new candidates to existing feature estimation entry
            if feature_estimate.has_candidates(design_param, fix_condition):
                new_candidates = feature_estimate.get_candidates(design_param, fix_condition)
                struct_index = feature_estimate.get_structural_vector_index(design_param, fix_condition)
                for candidate in new_candidates:
                    old_feature_estimate.add_candidate(design_param, fix_condition, candidate, struct_index)
                # add new candidates for child feature estimates
                for label in feature_estimate.get_child_labels():
                    if label not in fix_condition.get_labels():
                        child_feature_estimate = feature_estimate.get_child_feature_estimate(label)
                        old_child_feature_estimate = old_feature_estimate.get_child_feature_estimate(label)
                        new_candidates = child_feature_estimate.get_candidates(design_param, fix_condition)
                        struct_index = child_feature_estimate.get_structural_vector_index(design_param, fix_condition)
                        for candidate in new_candidates:
                            old_child_feature_estimate.add_candidate(design_param, fix_condition, candidate,
                                                                     struct_index)
        else:
            feature_evaluator = FeatureEvaluator(models)
            feature_estimator = FeatureEstimator(feature_evaluator)
            root_feature_estimate = FeatureEstimationResult('', feature_evaluator)
            fix_condition = ComponentFixCondition(root_feature_estimate)
            feature_estimate = feature_estimator.estimate(design_param, fix_condition, duplication_check,
                                                          num_candidate=num_candidate,
                                                          num_particle=num_particle,
                                                          max_iteration=max_iteration,
                                                          verbose=verbose)

            # register results to moldata
            if feature_estimate.has_candidates(design_param, fix_condition):
                self.add_feature_estimate(feature_estimate)
                for label in feature_estimate.get_child_labels():
                    if label not in fix_condition.get_labels():
                        subdata = self.get_subdata(label)
                        subdata.add_feature_estimate(feature_estimate.get_child_feature_estimate(label))

        return feature_estimate

    def get_ring_replacement(self, recursive=False):
        """Get the number of atoms in rings for ring replacement

        Args:
            recursive (bool): flag to make ring replacement for child moldata

        Returns:
            dict: mapping of atom and the number (mapping of label and mapping of atom and the number if recursive)
        """
        if not recursive:
            if self.mols_mask is None:
                molecules = self.mols
            else:
                molecules = [mol for mol, mask in zip(self.mols, self.mols_mask) if mask]
            return make_ring_replacement(molecules)
        replacements = dict()
        replacements[self.mol_label] = self.get_ring_replacement()
        for label in self.get_subdata_labels(recursive=recursive):
            replacements[label] = self.get_subdata(label).get_ring_replacement()
        return replacements

    def generate_molecule(self, models, design_param, fix_condition=None,
                          max_gen=0, max_solution=0, max_node=0, max_depth=0, beam_size=0,
                          renew=False, without_estimate=True, verbose=False):
        """Generate molecules satisfying estimated feature vector from a model. Generated molecules
         are stored in moldata.

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
            max_gen (int, optional): number of feature estimates for generation. Defaults to 0.
            max_solution (int, optional): maximum number of solutions to find. Defaults to 0.
            max_node (int, optional): maximum number of search tree nodes to search. Defaults to 0.
            max_depth (int, optional): maximum depth of iterative deepening. Defaults to 0.
            beam_size (int, optional): beam size of beam search. Defaults to 0.
            renew (bool): overwrite existing generated molecules
            without_estimate (bool): flag of generation without feature estimates
            verbose (bool, optional): flag of verbose message. Defaults to False.

        Returns:
            list: a lists of FeatureEstimationResult.Candidate
        """
        # check models
        for model in models:
            if isinstance(model.get_features(), MergedFeatureSet):
                if self != model.get_moldata():
                    logger.error('model has moldata not consistent with this moldata')
                    return None

        # check given design parameters
        if len(models) != len(design_param.get_target_values()):
            logger.error('invalid design parameter')
            return None
        target_properties = set()
        for model in models:
            if model.get_target_property() not in design_param.get_target_values():
                logger.error('target property %s in model is no in design parameter')
                return None
            if model.get_target_property() in target_properties:
                logger.error('duplicated target property in models')
                return None
            target_properties.add(model.get_target_property())

        # get estimated feature vectors
        feature_estimate = self.get_feature_estimate_by_models(models)
        if feature_estimate is None:
            if without_estimate:
                feature_evaluator = FeatureEvaluator(models)
                feature_estimate = FeatureEstimationResult(self.mol_label, feature_evaluator)
                if fix_condition is None:
                    fix_condition = ComponentFixCondition(feature_estimate)
                feature_evaluator.prepare_evaluation(design_param, fix_condition, True)
                # register results to moldata
                self.add_feature_estimate(feature_estimate)
            else:
                target_property_id = FeatureEvaluator.get_target_property_id_string(models)
                models_id = FeatureEvaluator.get_models_id_string(models)
                logger.error('no feature estimates for target property \'%s\' by model %s.',
                             target_property_id, models_id)
                return None

        # check feasibility of generation
        label = feature_estimate.get_label()
        evaluator = feature_estimate.get_evaluator()
        # check feasibility of without estimate
        if without_estimate:
            if not evaluator.has_only_online_feature() or \
                    not evaluator.has_single_target_label():
                logger.error('cannot generate molecule without feature estimation')
                return None
        if label == '' and len([lb for lb in evaluator.get_labels() if lb != '']) > 0:
            logger.error('feature estimate is infeasible for molecule generation')
            return None

        if fix_condition is None:
            fix_condition = ComponentFixCondition(feature_estimate)

        # get candidates not yet molecule generated
        if renew:
            # refresh generated molecules in candidate
            for candidate in feature_estimate.get_candidates(design_param, fix_condition):
                candidate.set_molecules(None)
                candidate.set_feature_extracted(False)

        target_candidates = []
        if without_estimate:
            # make feature estimate candidate without feature vector
            feature_dtype = evaluator.get_feature_dtype(with_mask=False)
            feature_range = evaluator.get_feature_range(with_mask=False)
            selection_mask = evaluator.get_selection_mask()
            evaluator.verbose = False
            null_candidate_id = FeatureEstimator.get_time_stamp_id()
            null_candidate = FeatureEstimationResult.Candidate(null_candidate_id, label, feature_estimate,
                                                               None, None,
                                                               feature_dtype, feature_range, selection_mask, 0)
            feature_estimate.add_candidate(design_param, fix_condition, null_candidate, [])
            target_candidates.append(null_candidate)
        else:
            # get feature estimate candidates up to max_gen
            for candidate in feature_estimate.get_candidates(design_param, fix_condition):
                if candidate.get_molecules() is None:
                    target_candidates.append(candidate)
                if 0 < max_gen <= len(target_candidates):
                    break

            if len(target_candidates) == 0:
                logger.error('no more candidate feature estimates for design params: %s',
                             design_param.get_short_id())
                return None

        # generate candidate molecule structure
        logger.info('generate molecules[{0}]: property {1} max_node={2} max_solution={3} max_depth={4} beam_size={5}'
                    .format(self.mol_label, design_param.get_target_values(), max_node, max_solution, max_depth, beam_size))
        design_param = feature_estimate.get_design_param(design_param.get_id())
        generator = MoleculeGenerator(label, evaluator)
        candidate_molecules = generator.generate_molecule(design_param, fix_condition,
                                                          feature_estimate,
                                                          target_candidates,
                                                          max_solution=max_solution,
                                                          max_node=max_node,
                                                          max_depth=max_depth,
                                                          beam_size=beam_size,
                                                          verbose=verbose)
        return candidate_molecules

    def estimate_and_generate(self, models, design_param, fix_condition=None, max_molecule=0, max_candidate=0,
                              num_candidate=1, num_particle=1000, max_iteration=1000,
                              label='', max_solution=0, max_node=0, max_depth=0, beam_size=0,
                              renew=False, verbose=False):
        """Estimate features and generate molecules.

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
            max_molecule (int, optional): number of molecules to obtain. Defaults to 0.
            max_candidate (int, optional): maximum number of candidate feature vector. Defaults to 0.
            num_candidate (int, optional): number of candidate feature vectors at a time. Defaults to 1.
            num_particle (int, optional): number of particles for optimization. Defaults to 1000.
            max_iteration (int, optional): maximum number of iteration for optimization. Defaults to 1000.
            label (str, optional): label of moldata component. Defaults to ''.
            max_solution (int, optional): maximum number of solutions to find. Defaults to 0.
            max_node (int, optional): maximum number of search tree nodes to search. Defaults to 0.
            max_depth (int, optional): maximum depth of iterative deepening. Defaults to 0.
            beam_size (int, optional): beam size of beam search. Defaults to 0.
            renew (bool): overwrite existing feature estimates
            verbose (bool, optional): flag of verbose message. Defaults to False.

        Returns:
            list: a lists of FeatureEstimationResult.Candidate
        """
        # check label
        if label != '' and label not in self.get_subdata_labels(recursive=True):
            logger.error('no sub component label:{0}'.format(label))
            return []

        if self.has_feature_estimate_by_models(models):
            old_feature_estimate = self.get_feature_estimate_by_models(models)
            if fix_condition is None:
                fix_condition = ComponentFixCondition(old_feature_estimate)
            # check existing candidates
            if old_feature_estimate.has_candidates(design_param, fix_condition):
                # clear candidates for design_param and fix_condition
                if renew:
                    self.clear_feature_estimate_candidates(models, design_param, fix_condition)
        elif fix_condition is None:
            feature_evaluator = FeatureEvaluator(models)
            root_feature_estimate = FeatureEstimationResult('', feature_evaluator)
            fix_condition = ComponentFixCondition(root_feature_estimate)

        molecule_candidate = []
        num_molecule = 0
        num_estimation = 0
        total_candidate = 0
        while True:
            if num_candidate == 1:
                logger.info('iterating estimate and generate ({0}/{1})'.
                            format(total_candidate + 1, max_candidate))
            else:
                logger.info('iterating estimate and generate ([{0}-{1}]/{2})'.
                            format(total_candidate + 1, total_candidate + num_candidate, max_candidate))
            # estimate feature
            feature_estimate = self.estimate_feature(models, design_param,
                                                     fix_condition=fix_condition, duplication_check=label,
                                                     num_candidate=num_candidate, num_particle=num_particle,
                                                     max_iteration=max_iteration,
                                                     verbose=verbose)
            if feature_estimate.has_candidates(design_param, fix_condition):
                num_estimation += 1
                total_candidate += len(feature_estimate.get_candidates(design_param, fix_condition))
            else:
                break

            # generate molecules
            subdata = self.get_subdata(label)
            candidates = subdata.generate_molecule(models, design_param, fix_condition=fix_condition,
                                                   max_solution=max_solution, max_node=max_node,
                                                   max_depth=max_depth, beam_size=beam_size,
                                                   without_estimate=False,
                                                   verbose=verbose)
            # count unique molecules
            molecules = set()
            for candidate in candidates:
                for molecule in candidate.get_molecules():
                    molecules.add(molecule.get_smiles())
            num_molecule += len(molecules)
            # add candidates
            molecule_candidate.extend(candidates)

            logger.info('{0} molecules in total'.format(num_molecule))
            logger.info('')

            if 0 < max_candidate <= total_candidate:
                break

            if 0 < max_molecule <= num_molecule:
                break

        return molecule_candidate

    def get_generated_molecule_summary(self, models, design_param, fix_condition=None,
                                       include_data=False,
                                       molobj=True, mols=True, smiles=True, property=True, estimates=True,
                                       features=False, generation_path=False):
        """Get a summary of generated molecules in a dataframe

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
            include_data (bool, optional): true to include training data matching to the feature vector.
                Default to False.
            molobj (bool, optional): true to include python Molecule object. Defaults to True
            mols (bool, optional): true to include molecule 2D image. Defaults to True,
            smiles (bool, optional): true to include SMILES of molecules. Defaults to True.
            property (bool, optional): true to include property of molecules. Defaults to True.
            estimates (bool, optional): true to include estimated target property. Default to True
            features (bool, optional): true to include feature vector values. Default to False
            generation_path (bool, optional): true to include generation path of the molecule. Default to False.

        Returns:
            DataFrame: a summary table of generated molecules
        """
        if not self.has_feature_estimate_by_models(models):
            logger.error('no feature estimation result specified by the models')
            return None

        feature_estimate = self.get_feature_estimate_by_models(models)
        evaluator = feature_estimate.get_evaluator()
        label = feature_estimate.get_label()
        if fix_condition is None:
            fix_condition = ComponentFixCondition(feature_estimate)

        candidates = feature_estimate.get_candidates(design_param, fix_condition)
        molecules = []
        for candidate in candidates:
            if not candidate.get_duplicate() and candidate.get_molecules() is not None:
                # extract the same features as moldata molecules if not yet extracted
                if not candidate.get_feature_extracted() and len(candidate.get_molecules()) > 0:
                    for c_features in evaluator.get_label_features_list(label):
                        c_features.extract_features(candidate.get_molecules())
                    candidate.set_feature_extracted(True)
                # add molecules to summary molecules
                molecules.extend(candidate.get_molecules())
        fit_candidate = dict()
        if include_data:
            # find fitting feature estimate candidate to a molecule
            fit_candidate = feature_estimate.choose_fit_candidate(design_param, fix_condition,
                                                                  self.get_mols())
            molecules.extend(fit_candidate.keys())
        # get properties from molecules
        if property:
            property_id_set = set()
            for molecule in molecules:
                for m_property_id in molecule.property_map.keys():
                    property_id_set.add(m_property_id)
        # make dataframe
        df_index = [m.get_id() for m in molecules]
        df = pd.DataFrame(index=df_index, data=df_index, columns=['ID'])
        if generation_path:
            df_path = pd.DataFrame(index=df_index, data=[m.get_generation_path() for m in molecules],
                                   columns=['gen_path'])
            df = df.join(df_path)
        if molobj:
            df_molecule = pd.DataFrame(index=df_index, data=molecules, columns=['molecule'])
            df = df.join(df_molecule['molecule'])
        if mols:
            smiles = [m.get_smiles() for m in molecules]
            df_smiles = pd.DataFrame(index=df_index, data=smiles, columns=['SMILES'])
            PandasTools.AddMoleculeColumnToFrame(df_smiles, smilesCol='SMILES', molCol='ROMol')
            df = df.join(df_smiles['ROMol'])
            for fix_label in fix_condition.get_labels():
                fix_label_prefix = fix_label + ':'
                fix_molecule = fix_condition.get_molecule(fix_label)
                smiles = [fix_molecule.get_smiles()] * len(df_index)
                df_smiles = pd.DataFrame(index=df_index, data=smiles, columns=['SMILES'])
                PandasTools.AddMoleculeColumnToFrame(df_smiles, smilesCol='SMILES', molCol=fix_label_prefix+'ROMol')
                df = df.join(df_smiles[fix_label_prefix+'ROMol'])
        if smiles:
            smiles = [m.get_smiles() for m in molecules]
            df_smiles = pd.DataFrame(index=df_index, data=smiles, columns=['SMILES'])
            df = df.join(df_smiles)
            for fix_label in fix_condition.get_labels():
                fix_label_prefix = fix_label + ':'
                fix_molecule = fix_condition.get_molecule(fix_label)
                smiles = [fix_molecule.get_smiles()] * len(df_index)
                df_smiles = pd.DataFrame(index=df_index, data=smiles, columns=[fix_label_prefix+'SMILES'])
                df = df.join(df_smiles)
        if property:
            for m_property_id in sorted(property_id_set):
                m_property = Property(m_property_id)
                properties = [m.get_property(m_property) for m in molecules]
                df_property = pd.DataFrame(index=df_index, data=properties, columns=[m_property_id])
                df = df.join(df_property)
        if estimates:
            target_values = design_param.get_target_values()
            df_estimates = evaluator.get_estimates_dataframe(label, target_values, molecules,
                                                             fit_candidate, fix_condition)
            df = df.join(df_estimates)
        if features:
            df_features = evaluator.get_features_dataframe(label, molecules, fit_candidate)
            df = df.join(df_features)
        # replace index with the array index of df
        df.index = np.array(range(len(molecules)))
        if generation_path:
            df = df.sort_values(['gen_path'], ascending=[True])
            # renumber index of dataframe by sorted result
            for index, df_index in enumerate(df.index.values):
                df.index.values[index] = index
        return df

    def print_properties(self):
        """Print registered properties.
        """
        print('properties:{0}'.format(self.properties.get_header_list()))

    def print_features(self):
        """Print registered feature sets.
        """
        print('feature set list:{0}'.format(self.mol_label))
        for index, feature in enumerate(self.features_list):
            print('  {0:d}: {1}'.format(index, feature.id))
        for label in self.get_subdata_labels():
            self.get_subdata(label).print_features()

    def print_merged_features(self):
        """Print registered merged feature sets.
        """
        print('merged feature set list:{0}'.format(self.mol_label))
        for index, feature in enumerate(self.merged_features_list):
            print('  {0:d}: {1}'.format(index, feature.id))
        for label in self.get_subdata_labels():
            self.get_subdata(label).print_merged_features()

    def print_regression_models(self):
        """Print all the registered regression model classified by a target property and a feature set.
        """
        print('regression model list:\'{0}\''.format(self.mol_label))
        rmodel_list = self.regression_model_list
        property_list = sorted(rmodel_list.keys())
        for property_id in property_list:
            print(' * target property: \'{0}\''.format(property_id))
            for features in self.features_list + self.merged_features_list:
                features_id = features.get_id()
                if features_id in rmodel_list[property_id]:
                    print('  + features: {0}'.format(features_id))
                    for index, snap_shot in enumerate(rmodel_list[property_id][features_id]):
                        model = snap_shot.get_model()
                        print('   {0:d}: R^2 score={2:.2f} cv_score={3:.2f} (+/- {4:.2f}) size={5:d}/{6:d} {1}'.
                              format(index, snap_shot.get_id(),
                                     model.get_score(), model.get_cv_score_mean(), model.get_cv_score_std(),
                                     model.get_vector_size(), model.get_features().get_vector_size()))
        for label in self.get_subdata_labels():
            self.get_subdata(label).print_regression_models()

    def print_feature_estimates(self):
        """Print all the registered feature estimates classified by a target property and a feature set.
        """
        print('feature estimate list:\'{0}\''.format(self.mol_label))
        for property_id in sorted(self.feature_estimate_list.keys()):
            property_ids = FeatureEvaluator.split_id_string(property_id)
            p_str = ''
            for p_id in property_ids:
                p_str += '\'{0}\','.format(p_id)
            print(' * target property: {0}'.format(p_str.rstrip(',')))
            for features_id in sorted(self.feature_estimate_list[property_id].keys()):
                features_ids = FeatureEvaluator.split_id_string(features_id)
                print('  + {0}'.format(features_ids[0]))
                for f_id in features_ids[1:]:
                    print('    {0}'.format(f_id))
                for index, feature_estimate in enumerate(self.feature_estimate_list[property_id][features_id]):
                    models_ids = FeatureEvaluator.split_id_string(feature_estimate.get_models_id())
                    print('   {0:d}: {1}'.format(index, models_ids[0]))
                    for m_id in models_ids[1:]:
                        print('      {0}'.format(m_id))
                    for (params_id, fix_cond_id) in feature_estimate.get_candidate_keys():
                        params = feature_estimate.get_design_param(params_id)
                        fix_cond = feature_estimate.get_fix_condition(fix_cond_id)
                        candidates = feature_estimate.get_candidates(params, fix_cond)
                        if len(fix_cond.get_labels()) > 0:
                            print('    - target value={0} params={1} fixed={2}'.
                                  format(params.get_target_values_id(),
                                         params.get_params_id(),
                                         fix_cond.get_id()))
                        else:
                            print('    - target value={0} params={1}'.
                                  format(params.get_target_values_id(),
                                         params.get_params_id()))
                        for c_index, candidate in enumerate(candidates):
                            print('     {0}: {1}'.format(c_index, candidate.to_string()))
        for label in self.get_subdata_labels():
            self.get_subdata(label).print_feature_estimates()

    def print_mol_info(self, id):
        """Print molecule information.

        Args:
            id (str): id of a molecule
        """
        self.get_mol(id).print(features_list=self.features_list)

    def print_mol_info_by_index(self, index):
        """Print molecule information.

        Args:
            index (int) index of a molecule in a list
        """
        self.get_mol_by_index(index).print(features_list=self.features_list)

    def get_dataframe(self, smiles=False, smarts=False, mols=False, property=False, features=None,
                      models=None, with_mask=False, readable=False, recursive=True):
        """Get a matrix of molecules and property and features as a pandas dataframe.

        Args:
            smiles (bool, optional): true to include SMILES of molecules. Defaults to False.
            smarts (bool, optional): true to include SMARTS of molecules. Defaults to False.
            mols (bool, optional): true to include molecule 2D image. Defaults to False.
            property (bool, optional): true to include property values. Defaults to False.
            features (FeatureSet, optional): specify a feature set to include feature values. Defaults to None.
            models (list): a list of regression models to include their estimate values. Defaults to None.
            with_mask (bool, optional): true to eliminate invalid molecules using mask of a feature set.
                Defaults to False.
            readable (bool, optional): print feature values in readable form
            recursive(bool, optional): flag for recursively collecting subdata for smiles, smarts, and mols.
                Defaults to True.

        Returns:
            DataFrame: a matrix of molecules and property and features
        """
        df_index = self.get_mol_index_list()
        df = pd.DataFrame(index=df_index)
        data_mask = self.mols_mask
        if mols:
            df_smiles = pd.DataFrame(index=df_index, data=self.get_safe_smiles_list(), columns=['SMILES'])
            PandasTools.AddMoleculeColumnToFrame(df_smiles, smilesCol='SMILES', molCol='ROMol')
            df = df.join(df_smiles['ROMol'])
            for label in self.get_subdata_labels(recursive=recursive):
                subdata = self.get_subdata(label)
                df_smiles = pd.DataFrame(index=df_index, data=subdata.get_safe_smiles_list(), columns=['SMILES'])
                PandasTools.AddMoleculeColumnToFrame(df_smiles, smilesCol='SMILES', molCol=label+':ROMol')
                df = df.join(df_smiles[label+':ROMol'])
                if with_mask:
                    data_mask = update_data_mask(data_mask, subdata.mols_mask)
        if smiles:
            df_smiles = pd.DataFrame(index=df_index, data=self.get_smiles_list(), columns=['SMILES'])
            df = df.join(df_smiles)
            for label in self.get_subdata_labels(recursive=recursive):
                subdata = self.get_subdata(label)
                df_smiles = pd.DataFrame(index=df_index, data=subdata.get_smiles_list(), columns=[label+':SMILES'])
                df = df.join(df_smiles)
                if with_mask:
                    data_mask = update_data_mask(data_mask, subdata.mols_mask)
        if smarts:
            df_smarts = pd.DataFrame(index=df_index, data=self.get_smarts_list(), columns=['SMARTS'])
            df = df.join(df_smarts)
            for label in self.get_subdata_labels(recursive=recursive):
                subdata = self.get_subdata(label)
                df_smarts = pd.DataFrame(index=df_index, data=subdata.get_smarts_list(), columns=[label+':SMARTS'])
                df = df.join(df_smarts)
                if with_mask:
                    data_mask = update_data_mask(data_mask, subdata.mols_mask)
        if property:
            df = df.join(self.get_property_vector())
        if features is not None:
            if isinstance(features, list):
                for f in features:
                    feature_vector = self.get_feature_vector(f.id, readable=readable)
                    feature_mask = self.get_feature_mask(f.id)
                    if feature_vector is not None:
                        df = df.join(feature_vector)
                        if with_mask:
                            data_mask = update_data_mask(data_mask, feature_mask)
            else:
                feature_vector = self.get_feature_vector(features.id, readable=readable)
                feature_mask = self.get_feature_mask(features.id)
                if feature_vector is not None:
                    df = df.join(feature_vector)
                    if with_mask:
                        data_mask = update_data_mask(data_mask, feature_mask)
        if models is not None:
            for model in models:
                if isinstance(model, tuple):
                    (m, f) = model
                    feature_vector = self.get_feature_vector(f.get_id())
                    estimate = m.predict(feature_vector)
                    df = df.join(estimate, rsuffix='{'+f.get_id()+'}')
                else:
                    feature_vector = self.get_feature_vector(model.get_features().get_id())
                    estimate = model.predict(feature_vector)
                    df = df.join(estimate)
        if with_mask and data_mask is not None:
            return df.loc[df.index[data_mask]]
        else:
            return df

    def write_csv(self, filename, smiles=True, smarts=False, property=True, features=None,
                  models=None, with_mask=False, readable=False):
        """Write a matrix of molecules and property and features to a file of CSV format.

        Args:
            filename (str): file name to save in
            smiles (bool, optional): true to include SMILES of molecules. Defaults to True.
            smarts (bool, optional): true to include SMARTS of molecules. Defaults to False.
            property (bool, optional): true to include property values. Defaults to True.
            features (FeatureSet, optional): specify a feature set to include feature values. Defaults to None.
            models (list): a list of regression models to include their estimate values. Defaults to None.
            with_mask (bool, optional): true to eliminate invalid molecules using mask of a feature set.
                Defaults to False.
            readable (bool, optional): print feature values in readable form
        """
        # get dataframe
        dataframe = self.get_dataframe(smiles=smiles, smarts=smarts, property=property, features=features,
                                       models=models, with_mask=with_mask, readable=readable)
        if dataframe is None:
            return
        dataframe.to_csv(filename)

    @classmethod
    def read_sdf(cls, filename):
        """Make a MolData object from an sdf file.

        Args:
            filename (str): a pth name of an SDF file

        Returns:
            MolData: a moldata object
        """
        # read from sdf file
        sdf = Chem.SDMolSupplier(filename)
        dataframe = PandasTools.LoadSDF(filename)

        # check index duplication
        if any(dataframe.index.duplicated()):
            dup_list = []
            for index in dataframe.index[dataframe.index.duplicated()]:
                dup_list.append(index)
            logger.error('duplicated indices in sdf file: %s', dup_list)
            return None

        molecules = []
        for index, mol in zip(dataframe.index, dataframe['ROMol']):
            if mol is None:
                logger.warning('skip reading molecule index=%s', index)
            else:
                molecules.append(SimpleMolecule(index, mol=mol, mol_block=sdf.GetItemText(index)))
        del dataframe['ROMol']
        # get properties, features from column
        property_list = []
        for prop_name in dataframe.columns:
            prop = Property(prop_name)
            property_list.append(prop)
            for mol, val in zip(molecules, dataframe[prop_name]):
                mol.set_property(prop, val)
        if len(molecules) > 0:
            moldata = MolData(molecules, PropertySet(property_list))
            return moldata
        else:
            logger.error('empty file:%s', filename)
            return None

    @classmethod
    def read_csv(cls, filename, smiles_col='SMILES', smarts_col='SMARTS'):
        """Make a MolData object from a csv file. Column of 0 of a CSV file is regarded as ids of molecules.
        There must be a column of 'SMILES' or 'SMARTS' to generate rdkit Mol objects. Other columns are
        treated as properties of a molecule.

        Args:
            filename (str): a path name of a CSV file
            smiles_col (str, optional): a name of column of SMILES. Defaults to 'SMILES'.
            smarts_col (str, optional): a name of column of SMARTS. Defaults to 'SMARTS'.

        Returns:
            MolData: a moldata object
        """
        # read from csv file
        dataframe = pd.read_csv(filename, index_col=0)

        # check index duplication
        if any(dataframe.index.duplicated()):
            dup_list = []
            for index in dataframe.index[dataframe.index.duplicated()]:
                dup_list.append(index)
            logger.error('duplicated indices in csv file: %s', dup_list)
            return None

        molecules = []
        if smiles_col in dataframe.columns:
            # make molecule from smiles
            for index, smiles in zip(dataframe.index, dataframe[smiles_col]):
                molecule = SimpleMolecule(index, smiles=smiles)
                if molecule.get_mol() is None:
                    logger.warning('skip reading molecule id=%s', index)
                else:
                    molecules.append(molecule)
        elif smarts_col in dataframe.columns:
            # make molecule from smarts
            for index, smarts in zip(dataframe.index, dataframe[smarts_col]):
                molecule = SimpleMolecule(index, smarts=smarts)
                if molecule.get_mol() is None:
                    logger.warning('skip reading molecule id=%s', index)
                else:
                    molecules.append(SimpleMolecule(index, smarts=smarts))
        else:
            logger.error('MolData.read_csv: no SMILES or SMARTS column')
            return None
        # get properties, features from column
        property_list = []
        for prop_name in dataframe.columns:
            if prop_name == smiles_col:
                del dataframe[prop_name]
                continue
            elif prop_name == smarts_col:
                del dataframe[prop_name]
                continue
            prop = Property(prop_name)
            property_list.append(prop)
            for mol, val in zip(molecules, dataframe[prop_name]):
                mol.set_property(prop, val)
        if len(molecules) > 0:
            moldata = MolData(molecules, PropertySet(property_list))
            return moldata
        else:
            logger.error('empty file:%s', filename)
            return None

    def plot_distribution(self, target_property, features, perplexity=30):
        """Plot data distribution for a target property.

        Args:
            target_property (str): target property
            features (MergedFeatureSet): feature set
            perplexity (int): perplexity for manifold TSNE
        """
        target = self.get_property_vector()[target_property].astype(float).values
        target_mask = list(map(lambda x: not np.isnan(x), target))
        if self.get_mols_mask() is not None:
            target_mask = [(x and y) for x, y in zip(target_mask, self.get_mols_mask())]
        color = target[target_mask]
        features_value = self.get_dataframe(features=features).values[target_mask]

        fig, ax = plt.subplots()
        ax.set_title("molecule distribution: property {0}".format(target_property))
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)
        y = tsne.fit_transform(features_value)
        sc = ax.scatter(y[:, 0], y[:, 1], vmin=color.min(), vmax=color.max(), c=color)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        fig.colorbar(sc)
        plt.show()

    def draw_candidate_molecules(self, models, design_param, fix_condition=None,
                                 max_draw=None, mols_per_row=10, use_svg=False):
        """Draw generated candidate molecules for target value.

        Args:
            models (list): a list of regression models
            design_param (DesignParam): molecule design parameter
            fix_condition (ComponentFixCondition, optional): fixed components. Defaults to None.
            max_draw (int, optional): a maximum number of candidates to draw. Defaults to None.
            mols_per_row(int, optional): a number of mols per row
            use_svg (bool, optional): use SVG drawing. Default to False.

        Returns:
            PIL: image of molecules
        """
        # get molecules
        candidate_molecules = self.get_generated_molecules(models, design_param, fix_condition)

        # draw molecules
        mols = []
        legends = []
        for molecules in candidate_molecules:
            for molecule in molecules:
                mols.append(molecule.get_mol())
                legends.append(molecule.get_id())
            if len(molecules) % mols_per_row > 0:
                for index in range(mols_per_row-len(molecules) % mols_per_row):
                    mols.append(Chem.MolFromSmiles(''))
                    legends.append('')
        return draw_rdkit_mols(mols, legends=legends, max_draw=max_draw,
                               mols_per_row=mols_per_row, use_svg=use_svg)

    @staticmethod
    def draw_features(features, max_draw=None, mols_per_row=None, sub_image_size=None,
                      legends=None, use_svg=False):
        """Draw sub-structure of molecules extracted by feature extraction.

        Args:
            features (FeatureSet): a feature set
            max_draw (int, optional): a maximum number of features to draw. Defaults to None.
            mols_per_row (int, optional): number of molecules to draw in a line. Defaults to None
            sub_image_size (tuple, optional): image size of each molecule. Defaults to None.
            legends (list, optional): title of each sub-structure. Defaults to None
            use_svg (bool, optional): use SVG drawing. Default to False.

        Returns:
            PIL: an image of features
        """
        feature_list = features.get_feature_list()
        mols = [f.mol for f in feature_list]
        if legends is None:
            legends = ['{0:d}:{1}'.format(index, f.id) for index, f in enumerate(feature_list)]
        return draw_rdkit_mols(mols, max_draw=max_draw, mols_per_row=mols_per_row, sub_image_size=sub_image_size,
                               legends=legends, use_svg=use_svg)

    @staticmethod
    def draw_molecules(molecules, max_draw=None, mols_per_row=None, sub_image_size=None,
                       legends=None, use_svg=False):
        """Draw molecules.

        Args:
            molecules (list): a list of Molecule objects
            max_draw (int, optional): a maximum number of features to draw. Defaults to None.
            mols_per_row (int, optional): number of molecules to draw in a line. Defaults to None.
            sub_image_size (tuple, optional): image size of each molecule. Defaults to None.
            legends (list): title of each sub-structure
            use_svg (bool, optional): use SVG drawing. Default to False.

        Returns:
            PIL: an image of molecules
        """
        mols = [m.get_mol() for m in molecules]
        if legends is None:
            legends = ['{0}'.format(m.get_id()) for m in molecules]
        return draw_rdkit_mols(mols, max_draw=max_draw, mols_per_row=mols_per_row, sub_image_size=sub_image_size,
                               legends=legends, use_svg=use_svg)

    @staticmethod
    def draw_mols(mols, max_draw=None, mols_per_row=None, sub_image_size=None,
                  legends=None, use_svg=False):
        """Draw molecules of rdkit Mol (a wrapper of rdkit Chem.Draw.MolsToGridImage).

        Args:
            mols (list): a list of rdkit Mol objects
            max_draw (int, optional): a maximum number of features to draw. Defaults to None.
            mols_per_row (int, optional): number of molecules to draw in a line. Defaults to None.
            sub_image_size (tuple, optional): image size of each molecule. Defaults to None.
            legends (list, optional): title of each sub-structure. Defaults to None.
            use_svg (bool, optional): use SVG drawing. Default to False.

        Returns:
            PIL: an image of molecules
        """
        return draw_rdkit_mols(mols, max_draw=max_draw, mols_per_row=mols_per_row, sub_image_size=sub_image_size,
                               legends=legends, use_svg=use_svg)

    def save(self, file):
        """Save pickle of moldata

        Args:
            file (str): file name
        """
        with open(file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        """Load pickle of moldata

        Args:
            file (str): file name

        Returns:
            MolData: moldata object
        """
        with open(file, 'rb') as f:
            moldata = pickle.load(f)
            return moldata
