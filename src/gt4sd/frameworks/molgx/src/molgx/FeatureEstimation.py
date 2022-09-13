# -*- coding:utf-8 -*-
"""
FeatureEstimation.py

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

from .FeatureExtraction import *
from .Utility import *

import numpy as np
import scipy as sp

import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# DesignParam: design parameters for the feature estimation and the generation
# -----------------------------------------------------------------------------

class DesignParam(object):
    """Parameters for feature estimation (target values and value range parameter for a feature vector).

    Attributes:
         target_values (dict): a mapping of target property and target value
         range_params (dict): parameters for generating value range of a feature vector
    """

    def __init__(self, target_values, range_params):
        """Constructor of DesignParam

        Args:
            target_values (dict): a mapping of target property and target value
            range_params (dict): parameters for generating value range of a feature vector
        """
        self.target_values = target_values
        self.range_params = range_params

    def get_id(self):
        """Get an id of design parameter

        Returns:
            tuple: tuple of parameter values
        """
        id_list = list(self.target_values.values())
        id_list.extend(list(self.range_params.values()))
        return tuple(id_list)

    def get_short_id(self):
        """Get a short id of design parameter for print

        Returns:
            tuple: tuple of parameter values
        """
        id_list = list(self.target_values.values())
        id_list.extend(list(self.range_params.values()))
        return tuple(id_list)

    def get_target_values_id(self):
        """Get an id of target values

        Returns:
            tuple: tuple of target values
        """
        return tuple(self.target_values.values())

    def get_params_id(self):
        """Get an id of range and generator parameters

        Returns:
            tuple: tuple of range pand generator parameters
        """
        id_list = list(self.range_params.values())
        return tuple(id_list)

    def get_target_values(self):
        """Get target values

        Returns:
            dict: target values
        """
        return self.target_values

    def get_range_params(self):
        """Get range parameters

        Returns:
            dict: range parameters
        """
        return self.range_params


# -----------------------------------------------------------------------------
# ComponentFixCondition: fixing molecule to label component position
# -----------------------------------------------------------------------------

class ComponentFixCondition(object):
    """Class for a set of molecules whose position is fixed in the feature estimate
    """

    def __init__(self, feature_estimate):
        """Constructor of ComponentFixCondition

        Args:
            feature_estimate (FeatureEstimationResult): context of fixing component
        """
        self.id = ''
        self.estimation_context = feature_estimate
        self.fixed_molecules = dict()
        self.fixed_vectors = dict()

    def get_id(self):
        """Get id of the condition for reference

        Returns:
            tuple: id of the condition
        """
        return self.id

    def get_mols_id(self):
        """Get id as molecule smiles

        Returns:
            dict: a mapping of label and smiles
        """
        mols_id = dict()
        for label in sorted(self.fixed_molecules.keys()):
            mols_id[label] = self.fixed_molecules[label].get_smiles()
        return mols_id

    def is_generated_in_context(self, label, molecule):
        """Check if a molecule is generated for label component in this estimation context

        Args:
            label (str): component label
            molecule (Molecule): a molecule

        Returns:
            bool: True if a molecule is generated in this context
        """
        if not isinstance(molecule, GeneratedMolecule):
            return False
        elif molecule.get_label() != label:
            return False
        elif molecule.get_vector_candidate().get_estimation_context().get_evaluator() != \
                self.estimation_context.get_evaluator():
            return False
        elif not molecule.get_vector_candidate().is_tight_feature_vector():
            return False
        else:
            return True

    def make_id(self):
        """Make id from fixed molecules

        Returns:
            tuple: id
        """
        id_list = []
        for label in self.estimation_context.get_child_labels():
            if label in self.fixed_molecules:
                molecule = self.fixed_molecules[label]
                # check if the molecule is generated in the same context
                if self.is_generated_in_context(label, molecule):
                    # all the member in the candidate are equivalent
                    # use candidate id
                    id_list.append('{0}:{1}'.format(label, molecule.get_vector_candidate().get_id()))
                else:
                    # use molecule smiles
                    id_list.append('{0}:{1}'.format(label, molecule.get_smiles()))
        return tuple(id_list)

    def get_labels(self):
        """Get a list of fixed components

        Returns:
            list: a list of labels
        """
        return sorted(list(self.fixed_molecules.keys()))

    def get_molecule(self, label):
        """Get a molecule of given label component

        Args:
            label (str): label of component

        Returns:
            SimpleMolecule: molecule at the label component
        """
        if label in self.fixed_molecules:
            return self.fixed_molecules[label]
        else:
            return None

    def get_feature_vector(self, label):
        """Get a feature vector of given label component

        Args:
            label (str): label of component

        Returns:
            np.ndarray: feature vector
        """
        if label in self.fixed_vectors:
            return self.fixed_vectors[label]
        else:
            return None

    def make_feature_vector(self, label, molecule):
        """Get a feature vector for a label component

        Args:
            label (str): label of component
            molecule (Molecule): a molecule

        Returns:
            np.ndarray: feature vector for a label component
        """
        if molecule is None:
            return None

        evaluator = self.estimation_context.get_evaluator()
        if self.is_generated_in_context(label, molecule):
            # get feature vector from candidate
            feature_vector = molecule.get_vector_candidate().get_feature_vector()
            return feature_vector
        else:
            # make feature vector from a molecule
            feature_vector = evaluator.get_merged_feature_vector_range(label, molecule)
            return feature_vector


# -----------------------------------------------------------------------------
# FeatureEstimationResult: a result of feature estimation in reverse problem
# -----------------------------------------------------------------------------

class FeatureEstimationResult(object):
    """Result of feature estimation for keeping all the found feature estimates and generate molecular structures

    Attributes:
        label (str): name of label component
        target_property_id (str): id as a first key of moldata dictionary
        features_id (str): id as a second key of moldata dictionary
        models_id (str): id as a third key of moldata dictionary
        evaluator (FeatureEvaluator): object for evaluating feature vector
        component_map (dict): a mapping of component label and child feature estimation result
        candidates_map (dict): a mapping of design parameter id and found feature estimates
        params_map (dict): a mapping of design parameter id and design parameter object
        fix_component_map (dict): a mapping of fixed component id and fixed component object
    """

    class Candidate(object):
        """Individual feature estimate and molecules generated from the feature estimate

        Attributes:
            id (str): unique id of candidate
            label (str): a label of component
            estimation_context (FeatureEstimationResult): context of parent feature estimation result
            whole_feature_vector (np.ndarray): a feature vector of the whole components
            feature_vector (np.ndarray): a feature vector as an array of intervals
            feature_dtype (np.ndarray): data type of feature vector values
            feature_range (np.ndarray): ranges of feature vector values
            selection_mask (list): selection mask of a feature vector
            score (float): score of feature estimation
            molecules (list): a list of generated SMILES
            duplicate (int): index of feature vector duplication in candidates
            feature_extracted (bool): flag of the completion of feature extraction
        """

        def __init__(self, id, label, context, whole_feature_vector,
                     feature_vector, feature_dtype, feature_range, selection_mask, score):
            """Constructor of FeatureEstimationResult.Candidate

            Args:
                id (str): id of candidate feature vector
                label (str): a label of component
                context (FeatureEstimationResult): context of feature estimation
                whole_feature_vector (np.ndarray, None): a feature vector of the whole components
                feature_vector (np.ndarray, None): a feature vector of this component
                feature_dtype (np.ndarray): data type of feature vector values
                feature_range (np.ndarray): ranges of feature vector values
                selection_mask (list): selection mask of a feature vector
                score (float): score of feature estimation
            """
            self.id = id
            self.label = label
            self.estimation_context = context
            self.whole_feature_vector = whole_feature_vector
            self.feature_vector = feature_vector
            self.feature_dtype = feature_dtype
            self.feature_range = feature_range
            self.selection_mask = selection_mask
            self.score = score
            self.molecules = None
            self.duplicate = -1
            self.feature_extracted = False

        def get_id(self):
            """Get id

            Returns:
                str: id string
            """
            return self.id

        def get_label(self):
            """Get a label of component

            Returns:
                str: a label
            """
            return self.label

        def get_mol_type(self):
            """Get a mol type of moldata

            Returns:
                MolType: moltype of target moldata
            """
            evaluator = self.estimation_context.get_evaluator()
            if evaluator.getmoldata() is not None:
                return evaluator.get_moldata().get_mol_type()
            else:
                return None

        def get_estimation_context(self):
            """Get a feature estimation result as a context

            Returns:
                FeatureEstimationResult: feature estimate
            """
            return self.estimation_context

        def get_whole_feature_vector(self):
            """Get a merged feature vector

            Returns:
                np.ndarray: a feature vector
            """
            if self.feature_vector is None:
                vector = np.zeros(self.estimation_context.get_evaluator().selection_size)
            else:
                vector = self.whole_feature_vector
            return vector

        def get_feature_vector(self):
            """Get a selected feature vector of label component

            Returns:
                np.ndarray: a feature vector
            """
            return self.feature_vector

        def get_feature_dtype(self, with_mask=True):
            """Get data types of a feature vector values

            Args:
                with_mask (bool): flag to apply selection mask

            Returns:
               np.ndarray: data types of a feature vector
            """
            if with_mask:
                return self.feature_dtype[self.selection_mask]
            else:
                return self.feature_dtype

        def get_feature_range(self, with_mask=True):
            """Get ranges of features vector values

            Args:
                with_mask (bool): flag to apply selection mask

            Returns:
                np.ndarray: ranges of a feature vector
            """
            if with_mask:
                return self.feature_range[self.selection_mask]
            else:
                return self.feature_range

        def get_selection_mask(self):
            """Get selection mask

            Returns:
                list: a selection mask
            """
            return self.selection_mask

        def get_score(self):
            """Get score of feature estimation

            Returns:
                float: score of feature estimation
            """
            return self.score

        def get_molecules(self):
            """Get a list of molecules

            Returns:
                list: a list of molecules
            """
            return self.molecules

        def set_molecules(self, molecules):
            """Set a list of generated molecules

            Args:
                 molecules (list, None): a list of molecules
            """
            self.molecules = molecules

        def get_duplicate(self):
            """Get duplication flag

            Returns:
                bool: duplication flag
            """
            return self.duplicate >= 0

        def set_duplicate(self, index):
            """Set duplication index

            Args:
                index (int): duplication index
            """
            self.duplicate = index

        def get_feature_extracted(self):
            """Get feature extracted flag

            Returns:
                bool: feature extracted flag
            """
            return self.feature_extracted

        def set_feature_extracted(self, extracted):
            """Set feature extracted flag

            Args:
                extracted (bool): feature extracted flag
            """
            self.feature_extracted = extracted

        def is_tight_feature_vector(self):
            """Check if the range for feature vector is tight

            Returns:
                bool: True if tight
            """
            if self.feature_vector is None:
                return False
            evaluator = self.estimation_context.get_evaluator()
            target_label = evaluator.get_target_label(self.label)
            index = 0
            for features in target_label.get_features_list():
                if not features.is_online_update():
                    if any(self.selection_mask[index:index+features.get_vector_size()]):
                        return False
            return np.all(self.feature_vector[:, 0] == self.feature_vector[:, 1])

        def to_string(self):
            """Get string representation

            Returns:
                str: sting representation
            """
            if self.feature_vector is None:
                rstr = '{0} no vector'.format(self.id)
            else:
                rstr = '{0} vector({1}) score={2:.3f}'.format(self.id, len(self.feature_vector), self.score)
            if self.duplicate >= 0:
                rstr += ' dup({0})'.format(self.duplicate)
            if self.molecules is not None:
                rstr += ': molecules={0}'.format(len(self.molecules))
            return rstr

    def __init__(self, label, evaluator):
        """Constructor of FeatureEstimationResult.

        Args:
            label (str): label component name
            evaluator (FeatureEvaluator): feature evaluator
        """
        self.label = label
        self.target_property_id = evaluator.get_target_property_id()
        self.features_id = evaluator.get_features_id()
        self.models_id = evaluator.get_models_id()
        self.evaluator = evaluator
        self.component_map = dict()
        self.candidates_map = defaultdict(list)
        self.params_map = dict()
        self.struct_index_map = defaultdict(list)
        self.fix_component_map = dict()

    def choose_fit_candidate(self, design_param, fix_condition, molecules):
        """Choose candidate whose feature vector fit to a molecule.

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate
            molecules (list): a list of molecules

        Returns:
            dict: a mapping from molecule to fitting candidate
        """
        fit_candidate = dict()
        candidates = self.get_candidates(design_param, fix_condition)
        for candidate in candidates:
            feature_vector = candidate.get_feature_vector()
            if feature_vector is None:
                continue
            used_smiles = set()
            for molecule in molecules:
                smiles = molecule.get_smiles()
                if smiles in used_smiles:
                    continue
                else:
                    used_smiles.add(smiles)
                mol_fv = self.evaluator.get_merged_feature_vector(self.label, molecule, candidate)
                match = True
                for index in range(len(feature_vector)):
                    if not feature_vector[index][0] <= mol_fv[index] <= feature_vector[index][1]:
                        match = False
                        break
                if match:
                    fit_candidate[molecule] = candidate
        return fit_candidate

    def get_label(self):
        """Get a name of label component

        Returns:
            str: a label component name
        """
        return self.label

    def get_target_property_id(self):
        """Get target properties as id

        Returns:
            str: id of target property
        """
        return self.evaluator.get_target_property_id()

    def get_features_id(self):
        """Get merged features of models as id

        Returns:
            str: id of merged features
        """
        return self.evaluator.get_features_id()

    def get_models_id(self):
        """Get regression models as id

        Returns:
            str: id of regression models
        """
        return self.evaluator.get_models_id()

    def get_evaluator(self):
        """Get feature evaluator object

        Returns:
            FeatureEvaluator: feature evaluator
        """
        return self.evaluator

    def get_candidate_keys(self):
        """Get a list of keys for candidates

        Returns:
            list: a list of keys for candidates
        """
        return list(self.candidates_map.keys())

    def get_design_param(self, id):
        """Get a design param object

        Args:
            id (tuple): id of design param

        Returns:
            DesignParam: design param
        """
        if id in self.params_map:
            return self.params_map[id]
        else:
            return None

    def get_design_params(self):
        """Get a list of design parameter values

        Returns:
            list: a list of design parameters
        """
        return list(self.params_map.values())

    def get_fix_condition(self, id):
        """Get a fix condition object

        Args:
            id (tuple): id of fix condition

        Returns:
            ComponentFixCondition: fix condition
        """
        if id in self.fix_component_map:
            return self.fix_component_map[id]
        else:
            return None

    def get_fix_conditions(self):
        """Get a list of fixed component values

        Returns:
            list: a list of fixed components
        """
        return list(self.fix_component_map.values())

    def get_child_labels(self):
        """Get a list of labels of child feature estimate

        Returns:
            list: a list of labels
        """
        return sorted(list(self.component_map.keys()))

    def add_child_feature_estimate(self, label, feature_estimate):
        """Add a feature estimation result for a component with a label

        Args:
            label (str): a label of component
            feature_estimate (FeatureEstimationResult): a feature estimation result for a component
        """
        self.component_map[label] = feature_estimate

    def get_child_feature_estimate(self, label):
        """Get a feature estimation result of a given label

        Args:
            label (str): a label of component

        Returns:
            FeatureEstimationResult: a child feature estimation result
        """
        if label in self.component_map:
            return self.component_map[label]
        else:
            return None

    def add_candidate(self, design_param, fix_condition, candidate, struct_index):
        """Add a candidate of feature estimation

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate
            candidate (FeatureEstimationResult.Candidate): a candidate feature estimation
            struct_index (list): a list of structural vector index
        """
        self.params_map[design_param.get_id()] = design_param
        self.fix_component_map[fix_condition.get_id()] = fix_condition
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        candidate_list = self.candidates_map[candidate_key]
        if candidate.get_feature_vector() is not None:
            # check duplication
            feature_vector = candidate.get_feature_vector()
            for index, candidate0 in enumerate(candidate_list):
                feature_vector0 = candidate0.get_feature_vector()
                if feature_vector0 is not None:
                    if np.all(feature_vector0[struct_index] == feature_vector[struct_index]):
                        candidate.set_duplicate(index)
                        break
        candidate_list.append(candidate)
        self.struct_index_map[candidate_key] = struct_index

    def get_candidate(self, design_param, fix_condition, id):
        """Get candidate of feature estimation

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate
            id (str): id of candidate
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        if candidate_key in self.candidates_map:
            candidate_list = self.candidates_map[candidate_key]
            for index, candidate in enumerate(candidate_list):
                if id == candidate.id:
                    return candidate_list[index]
            logger.error('get_candidate: no candidate id: {0}'.format(id))
        else:
            logger.error('get_candidate: no candidate entry for the key: {0}'.format(candidate_key))
        return None

    def get_candidate_by_index(self, design_param, fix_condition, index):
        """Remove candidate of feature estimation

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate
            index(int): index of candidate
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        if candidate_key in self.candidates_map:
            candidate_list = self.candidates_map[candidate_key]
            if 0 <= index < len(candidate_list):
                return candidate_list[index]
            else:
                logger.error('get_candidate: no candidate index: {0}'.format(index))
        else:
            logger.error('get_candidate: no candidate entry for the key: {0}'.format(candidate_key))
        return None

    def remove_candidate(self, design_param, fix_condition, id):
        """Remove candidate of feature estimation

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate
            id (str): id of candidate
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        if candidate_key in self.candidates_map:
            candidate_list = self.candidates_map[candidate_key]
            struct_index = self.struct_index_map[candidate_key]
            found_index = -1
            for index, candidate in enumerate(candidate_list):
                if id == candidate.id:
                    found_index = index
                    break
            if found_index >= 0:
                del candidate_list[found_index]
                # update duplicate flag
                for candidate in candidate_list[found_index:]:
                    candidate.set_duplicate(-1)
                    feature_vector = candidate.get_feature_vector()
                    for c_index, candidate0 in enumerate(candidate_list):
                        if candidate0.id == candidate.id:
                            break
                        feature_vector0 = candidate0.get_feature_vector()
                        if feature_vector is not None and feature_vector0 is not None:
                            if np.all(feature_vector0[struct_index] == feature_vector[struct_index]):
                                candidate.set_duplicate(c_index)
                                break
            else:
                logger.error('remove_candidate: no candidate id: {0}'.format(id))
        else:
            logger.error('remove_candidate: no candidate entry for the key: {0}'.format(candidate_key))

    def remove_candidate_by_index(self, design_param, fix_condition, index):
        """Remove candidate of feature estimation

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate
            index(int): index of candidate
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        if candidate_key in self.candidates_map:
            candidate_list = self.candidates_map[candidate_key]
            struct_index = self.struct_index_map[candidate_key]
            if 0 <= index < len(candidate_list):
                del candidate_list[index]
                # update duplicate flag
                for candidate in candidate_list[index:]:
                    candidate.set_duplicate(-1)
                    feature_vector = candidate.get_feature_vector()
                    for c_index, candidate0 in enumerate(candidate_list):
                        if candidate0.id == candidate.id:
                            break
                        feature_vector0 = candidate0.get_feature_vector()
                        if feature_vector is not None and feature_vector0 is not None:
                            if np.all(feature_vector0[struct_index] == feature_vector[struct_index]):
                                candidate.set_duplicate(c_index)
                                break
            else:
                logger.error('remove_candidate: no candidate index: {0}'.format(index))
        else:
            logger.error('remove_candidate: no candidate entry for the key: {0}'.format(candidate_key))

    def clear_candidates(self, design_param, fix_condition):
        """Clear candidates of feature estimation

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        if candidate_key in self.candidates_map:
            self.candidates_map[candidate_key] = []
        else:
            logger.error('clear_candidate: no candidate for the key: {0}'.format(candidate_key))

    def has_candidates(self, design_param, fix_condition):
        """Check if candidates are registered

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate

        Returns:
            bool: true if the key of candidates is registered
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        return candidate_key in self.candidates_map

    def get_candidates(self, design_param, fix_condition):
        """Get a list of candidates for given design parameter

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate

        Returns:
            list: a list of candidate feature estimates
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        if candidate_key in self.candidates_map:
            return self.candidates_map[candidate_key]
        else:
            logger.error('get_candidates: no candidate for the key: {0}'.format(candidate_key))
            return None

    def get_structural_vector_index(self, design_param, fix_condition):
        """Get a list of structural vector index

        Args:
            design_param (DesignParam): design parameter as key of candidate
            fix_condition (ComponentFixCondition): fixed components as a key of candidate

        Returns:
            list: a list of structural vector index
        """
        candidate_key = (design_param.get_id(), fix_condition.get_id())
        if candidate_key in self.struct_index_map:
            return self.struct_index_map[candidate_key]
        else:
            logger.error('get_structural_vector_index: no entry for the key: {0}'.format(candidate_key))
            return None

    def get_features_list(self, label=None):
        """Get a list of feature sets for a feature vector

        Args:
            label (str, optional): label component name. Defaults to None.

        Returns:
            list: a list of feature sets
        """
        if label is None:
            label = self.label
        target_label = self.evaluator.get_target_label(label)
        if target_label is None:
            return self.evaluator.get_features_list()
        else:
            return target_label.get_features_list()

    def get_feature_list(self, label=None):
        """Get a list of features for a feature vector

        Args:
            label (str, optional): label component name. Defaults to None.

        Returns:
            list: a list of features
        """
        if label is None:
            label = self.label
        target_label = self.evaluator.get_target_label(label)
        if target_label is None:
            return self.evaluator.get_feature_list()
        else:
            return target_label.get_feature_list()

    def get_selection_mask(self, label=None):
        """Get a selection mask for a feature vector

        Args:
            label (str, optional): label component name. Defaults to None.

        Returns:
            list: a selection mask
        """
        if label is None:
            label = self.label
        target_label = self.evaluator.get_target_label(label)
        if target_label is None:
            return self.evaluator.get_selection_mask()
        else:
            return target_label.get_selection_mask()

    def to_string(self, design_params=None):
        """Get string format of a feature estimation result.

        Returns:
            str: string format of a feature estimation result
        """
        rstr = 'estimated features for label component \'{0}\'\n'.format(self.label)
        for (params_id, fix_cond_id), candidates in self.candidates_map.items():
            params = self.params_map[params_id]
            fix_cond = self.fix_component_map[fix_cond_id]
            if design_params is not None and design_params.get_id() != params.get_id():
                continue
            rstr += ' * target value={0} params={1}\n'.format(params.get_target_values_id(), params.get_params_id())
            if len(fix_cond.get_labels()) > 0:
                rstr += ' * fixed molecule={0}\n'.format(fix_cond.get_mols_id())
            for index, candidate in enumerate(candidates):
                feature_vector = candidate.get_feature_vector()
                feature_dtype = candidate.get_feature_dtype()
                labels = self.get_evaluator().get_labels()
                if self.label != '':
                    labels = [self.label]
                rstr += '  + feature_vector[{0}]:'.format(index)
                label_index = 0
                for label in labels:
                    rstr += ' [\'{0}\']:'.format(label) if label != '' else ''
                    rstr += '{'
                    if feature_vector is None:
                        rstr += 'None'
                    else:
                        label_feature_list = self.get_features_list(label)
                        label_selection_mask = self.get_selection_mask(label)
                        vindex = 0
                        data_index = 0
                        for feature in label_feature_list:
                            headers = feature.get_header_list()
                            for vidx in range(feature.get_vector_size()):
                                if label_selection_mask[vindex+vidx]:
                                    coef = feature_vector[label_index+data_index]
                                    if coef[0] == coef[1]:
                                        if feature_dtype[label_index+data_index] == int:
                                            rstr += '{0}:[{1:d}] '.format(headers[vidx], int(coef[0]))
                                        else:
                                            rstr += '{0}:[{1:.2f}] '.format(headers[vidx], coef[0])

                                    else:
                                        if feature_dtype[label_index+data_index] == int:
                                            rstr += '{0}:[{1:d},{2:d}] '.format(headers[vidx], int(coef[0]),
                                                                                int(coef[1]))
                                        else:
                                            rstr += '{0}:[{1:.2f},{2:.2f}] '.format(headers[vidx], coef[0], coef[1])
                                    data_index += 1
                            vindex += feature.get_vector_size()
                    rstr = rstr.rstrip() + '}'
                    label_index += data_index
                rstr += '\n'
        return rstr.rstrip('\n')

    def print(self, design_params=None):
        """Print a feature estimation result.
        """
        print(self.to_string(design_params=design_params))

    def get_dataframe(self, design_params=None):
        """Get a dataframe of estimated feature vectors

        Returns:
            DataFrame: a dataframe of estimated feature vectors
        """
        feature_vectors = []
        feature_dtype = None
        for (params_id, fix_cond_id), candidates in self.candidates_map.items():
            params = self.params_map[params_id]
            if design_params is not None and design_params.get_id() != params.get_id():
                continue
            for index, candidate in enumerate(candidates):
                if candidate.get_feature_vector() is not None:
                    feature_vectors.append(candidate.get_feature_vector())
                    feature_dtype = candidate.get_feature_dtype()

        values = dict()
        index = ['{0}'.format(idx) for idx in range(len(feature_vectors))]
        labels = self.get_evaluator().get_labels()
        if self.label != '':
            labels = [self.label]
        label_index = 0
        for label in labels:
            prefix = '{0}:'.format(label) if label != '' else ''
            label_feature_list = self.get_features_list(label)
            label_selection_mask = self.get_selection_mask(label)
            vindex = 0
            data_index = 0
            for feature in label_feature_list:
                headers = feature.get_header_list()
                for vidx in range(feature.get_vector_size()):
                    if label_selection_mask[vindex+vidx]:
                        header = prefix + headers[vidx]
                        values[header] = []
                        for fv in feature_vectors:
                            coef = fv[label_index+data_index]
                            if coef[0] == coef[1]:
                                if feature_dtype[label_index+data_index] == int:
                                    values[header].append('{0:d}'.format(int(coef[0])))
                                else:
                                    values[header].append('{0:.2f}'.format(coef[0]))
                            else:
                                if feature_dtype[label_index+data_index] == int:
                                    values[header].append('[{0:d},{1:d}]'.format(int(coef[0]), int(coef[1])))
                                else:
                                    values[header].append('[{0:.2f},{1:.2f}]'.format(coef[0], coef[1]))
                        data_index += 1
                vindex += feature.get_vector_size()
            label_index += data_index
        df = pd.DataFrame(values, index=index)
        return df

# -----------------------------------------------------------------------------
# FeatureEstimator: feature estimator in reverse problem
# -----------------------------------------------------------------------------


class FeatureEstimator(object):
    """Estimating feature vector obtaining target value from given regression model.

    Attributes:
        evaluator (FeatureEvaluator): a evaluator of feature vectors
    """

    def __init__(self, evaluator):
        """Constructor of FeatureEstimator.

        Args:
            evaluator (FeatureEvaluator): evaluator of feature vector
        """
        self.evaluator = evaluator

    def estimate(self, design_param, fix_condition, duplication_check,
                 num_candidate=10, old_estimates=[], num_particle=1000, max_iteration=1000, verbose=True):
        """Estimate feature vector obtaining target values.

        Args:
            design_param (DesignParam): a design parameter
            fix_condition (ComponentFixCondition): fixed components
            duplication_check (str): label of component for duplication check.
            num_candidate (int, optional): a number of feature estimates to get. Default to 10.
            old_estimates (list, optional): a list existing feature estimates. Default to [].
            num_particle (int, optional): number of particles for optimization. Default to 1000.
            max_iteration (int, optional): maximum number of iteration for optimization. Default to 1000.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            FeatureEstimationResult: a root feature estimation result
        """
        self.evaluator.verbose = verbose

        # get target_values and range parameters
        target_values = design_param.get_target_values()
        range_params = design_param.get_range_params()
        max_atom = range_params['max_atom']
        max_ring = range_params['max_ring']
        extend_solution = range_params['extend_solution']
        sigma_ratio = range_params['sigma_ratio']
        count_tolerance = range_params['count_tolerance']
        count_min_ratio = range_params['count_min_ratio']
        count_max_ratio = range_params['count_max_ratio']
        prediction_error = range_params['prediction_error']

        if verbose:
            if len(fix_condition.get_labels()) == 0:
                print('estimate feature vector for {0}'.format(target_values))
            else:
                print('estimate feature vector for {0} with {1} fixed'.
                    format(target_values, fix_condition.get_labels()))
            logger.info('  with params max_atom:{0} max_ring:{1}, extend_solution:{2}'.
                  format(max_atom, max_ring, extend_solution))
            logger.info('    sigma_ratio:{0}, count_tolerance:{1} count_min_ration:{2} count_max_ratio:{3}'.
                  format(sigma_ratio, count_tolerance, count_min_ratio, count_max_ratio))
            for target_property in target_values:
                target_model = self.evaluator.get_target_model(target_property)
                logger.info('\'{0}\': {1}'.format(target_property, target_model.get_model().get_features().get_id()))
                logger.info('  model: {0}'.format(target_model.get_model().get_id()))

        # estimate features by optimizing target value
        feature_vector = self.evaluator.get_feature_vector()
        feature_dtype = self.evaluator.get_feature_dtype(with_mask=False)
        selection_mask = self.evaluator.get_selection_mask()
        feature_range = self.evaluator.prepare_evaluation(design_param, fix_condition)

        # get structural feature vector index for duplication check
        root_struct_index = self.evaluator.prepare_root_struct_index()
        if duplication_check != '':
            target_label = self.evaluator.get_target_label(duplication_check)
            selection_slice = target_label.get_selection_slice()
            struct_index = [s_index + selection_slice.start for s_index in target_label.get_structural_vector_index()]
        else:
            struct_index = root_struct_index

        # make optimizer
        optimizer = ParticleSwarmOptimization(self.evaluator, feature_vector, feature_range[selection_mask],
                                              feature_dtype[selection_mask], struct_index, fix_condition=fix_condition,
                                              prediction_error=prediction_error, extend_solution=extend_solution,
                                              num_particle=num_particle, iteration=max_iteration)

        # get optimized feature vectors
        feature_vectors, feature_vector_ranges, scores = \
            optimizer.optimize(target_values, num_candidate, old_estimates, verbose=verbose)

        id_string = self.get_time_stamp_id()
        cindex = len(old_estimates)

        # make feature estimate of the whole feature vector
        root_feature_estimate = FeatureEstimationResult('', self.evaluator)
        for index, (feature_vector, feature_vector_range, score) in \
                enumerate(zip(feature_vectors, feature_vector_ranges, scores)):
            id = '{0}V{1}'.format(id_string, cindex+index)
            candidate = FeatureEstimationResult.Candidate(id,
                                                          '',
                                                          root_feature_estimate,
                                                          feature_vector,
                                                          feature_vector_range,
                                                          feature_dtype,
                                                          feature_range,
                                                          selection_mask,
                                                          score)
            root_feature_estimate.add_candidate(design_param, fix_condition, candidate, root_struct_index)

        # make feature estimate of label component other than ''
        fixed_labels = fix_condition.get_labels()
        for label in self.evaluator.get_labels():
            if label == '':
                continue
            if label in fixed_labels:
                continue
            target_label = self.evaluator.get_target_label(label)
            vector_slice = target_label.get_vector_slice()
            selection_slice = target_label.get_selection_slice()
            label_feature_dtype = feature_dtype[vector_slice]
            label_feature_range = feature_range[vector_slice]
            label_selection_mask = selection_mask[vector_slice]
            label_struct_index = target_label.get_structural_vector_index()
            feature_estimate = FeatureEstimationResult(label, self.evaluator)
            # add feature vectors to feature estimate result
            for index, (feature_vector_range, score) in enumerate(zip(feature_vector_ranges, scores)):
                id = '{0}V{1}'.format(id_string, cindex + index)
                candidate = FeatureEstimationResult.Candidate(id,
                                                              label,
                                                              feature_estimate,
                                                              feature_vector,
                                                              feature_vector_range[selection_slice],
                                                              label_feature_dtype,
                                                              label_feature_range,
                                                              label_selection_mask,
                                                              score)
                feature_estimate.add_candidate(design_param, fix_condition, candidate, label_struct_index)
            root_feature_estimate.add_child_feature_estimate(label, feature_estimate)

        return root_feature_estimate

    @staticmethod
    def get_time_stamp_id():
        """Get time stamp as id of generated molecule

        Returns:
            str: time stamp
        """
        now = datetime.today()
        now_str = '{0:04d}{1:02d}{2:02d}{3:02d}{4:02d}{5:02d}{6:03d}'. \
            format(now.year, now.month, now.day, now.hour, now.minute, now.second, int(now.microsecond/1000))
        return now_str


# -----------------------------------------------------------------------------
# FeatureEvaluator: feature evaluator in reverse problem
# -----------------------------------------------------------------------------


class FeatureEvaluator(object):
    """Class for evaluating given merged feature vectors by models and labels for a multi-target
    feature estimation.

    Attributes:
        moldata (MolData): a moldata object
        target_models (dict): a mapping of target property and TargetModel object
        target_labels (dict): a mapping of label component and TargetLabel object
        labels (list): a list of label components
        vector_size (int): a size of the merged feature vector
        features_list (list): a list of feature sets for the merged feature vector
        feature_list (list): a list of features for the merged feature vectors
        selection_mask (list): a selection mask for the merged feature vectors
        feature_vector (matrix): an actual data of moldata for the merged feature vectors
        feature_dtype (array): an array of feature data type of the merged feature vectors
        feature_range (matrix): min/max range of feature vector element value
        verbose (bool): flag of verbose message
    """

    class TargetModel(object):
        """Class for retrieving an original feature vector for a regression model from a merged
        feature vector for multi-target feature estimation.

        Attributes:
            model (RegressionModel): a regression model
            vector_size (int): a vector size of original feature vector of the regression model
            selection_size (int): a vector size of selected feature vector
            labels (list): a list of labels ('' for merged feature set)
            feature_list_map (dict): a mapping of label and a list of features in the label
            features_list_map (dict): a mapping of label and a list of feature sets in the label
            features_list_slice_map (dict): a mapping of label and a slice of feature sets in the whole feature sets
            features_map (dict): a mapping of label and a mapping of feature ids and features in the label
            features_slice_map (dict): a mapping of label and a mapping of feature set id and its slice
                in a feature vector
            selection_mask_map (dict): a mapping of label and a mapping of feature set id and its selection mask
            feature_dtype_map (dict): a mapping of label and an array of feature types of a feature vector
            feature_range_map (dict): a mapping of label and an array of feature value range of a feture vector
            selection_slice_map (dict): a mapping of label and a mapping of feature set id and its slice of
                selected features
            selection_offset_map (dict): a mapping of label and an offset index in merged feature vector
            selection_ref_map (dict): a mapping of label and an index of selected feature vector
            feature_vector_slice_map (dict): a mapping of label and a slice of a feature vector
                in the whole feature vector
        """

        def __init__(self, model):
            """Constructor of FeatureEvaluator.TargetModel class

            Args:
                model (RegressionModel): a regression model
            """
            self.model = model
            self.vector_size = 0
            self.selection_size = 0
            self.labels = []
            self.feature_list_map = dict()
            self.features_list_map = dict()
            self.features_list_slice_map = dict()
            self.features_map = dict()
            self.features_slice_map = dict()
            self.selection_mask_map = dict()
            self.feature_dtype_map = dict()
            self.feature_range_map = dict()
            self.selection_slice_map = dict()
            self.selection_offset_map = dict()
            self.selection_ref_map = dict()
            self.feature_vector_slice_map = dict()

            # initialize dictionaries for labels
            self.initialize()

        def get_model(self):
            """Get a regression model

            Returns:
                RegressionModel: a regression model
            """
            return self.model

        def get_target_property(self):
            """Get a target property name

            Returns:
                str: a property name
            """
            return self.model.get_target_property()

        def get_vector_size(self):
            """Get a feature vector size for the regression model

            Returns:
                int: feature vector size
            """
            return self.vector_size

        def get_selection_size(self):
            """Get a selected feature vector size

            Returns:
                int: feature vector size
            """
            return self.selection_size

        def get_labels(self):
            """Get a list of labels

            Returns:
                list: a list of labels
            """
            return self.labels

        def get_feature_list(self, label):
            """Get a list of features in the label.

            Args:
                label (str): label name

            Returns:
                list: a list of features
            """
            return self.feature_list_map[label]

        def get_features_list(self, label):
            """Get a list of feature sets in the label

            Args:
                label (str): label name

            Returns:
                list: a list of feature sets
            """
            return self.features_list_map[label]

        def get_features(self, label):
            """Get a mapping of feature set ids and feature set object in the label

            Args:
                label (str): label name

            Returns:
                dict: a mapping of feature set id and feature set object
            """
            return self.features_map[label]

        def get_features_slice(self, label):
            """Get a mapping of feature set ids and its slice in a feature vector

            Args:
                label (str): label name

            Returns:
                dict: a mapping of feature set id and its slice
            """
            return self.features_slice_map[label]

        def get_selection_mask(self, label):
            """Get a feature selection mask of a feature vector in the label

            Args:
                label (str): label name

            Returns:
                list: a feature selection mask
            """
            return self.selection_mask_map[label]

        def get_feature_vector_slice(self, label):
            """Get a slice of a feature vector of the label

            Args:
                label (str): label name

            Returns:
                slice: a slice of feature vector
            """
            return self.feature_vector_slice_map[label]

        def get_selection_offset(self, label):
            """Get an offset of feature vector index in the merged feature vector

            Args:
                label (str): label name

            Returns:
                int: an offset of feature vector index
            """
            return self.selection_offset_map[label]

        def set_selection_offset(self, label, value):
            """Get an offset of feature vector index in the merged feature vector

            Args:
                label (str): label name
                value (int): an offset of feature vector index
            """
            self.selection_offset_map[label] = value

        def get_selection_ref(self, label):
            """Get an index of selected feature vector corresponding to the original feature vector

            Args:
                label (str): label name

            Returns:
                array: an index of selected feature vector
            """
            return self.selection_ref_map[label]

        def get_selection_slice(self, label):
            """Get a slice of selected feature vector for the label

            Args:
                label (str): label name

            Returns:
                slice: a slice of selected feature vector
            """
            return self.selection_slice_map[label]

        def get_ring_range(self, label):
            """Get a range of ring number for the label

            Args:
                label (str): label name

            Returns:
                array: a range of ring number
            """
            return self.ring_range_map[label]

        def initialize(self):
            """Initialize FeatureEvaluator.TargetModel class
            """
            # make a map of label and features list, feature vectors ...
            model_features = self.model.get_features()
            if isinstance(model_features, MergedFeatureSet):
                mfeatures = model_features
                self.labels.append('')
                # set slices of merged feature set
                self.feature_list_map[''] = mfeatures.get_feature_list()
                self.features_list_map[''] = mfeatures.get_features_list()
                selection_mask = None
                if self.model.is_feature_selected():
                    selection_mask = self.model.get_selection_mask()
                if selection_mask is None:
                    selection_mask = [True] * mfeatures.get_vector_size()
                self.selection_mask_map[''] = selection_mask
                self.vector_size += len(selection_mask)
                self.selection_size += sum(selection_mask)
                self.selection_ref_map[''] = np.zeros(sum(selection_mask), dtype=int)
                self.feature_vector_slice_map[''] = slice(0, mfeatures.get_vector_size())
                # make slices for each feature set in merged feature set
                features_id_map = dict()
                features_slice_map = dict()
                selection_slice_map = dict()
                index = 0
                selection_index = 0
                for features in mfeatures.get_features_list():
                    feature_vector_slice = mfeatures.get_feature_vector_slice(features)
                    features_id_map[features.id] = features
                    size = 0
                    selection_size = 0
                    for mask in selection_mask[feature_vector_slice]:
                        selection_size += 1 if mask else 0
                        size += 1
                    features_slice_map[features.id] = slice(index, index + features.get_vector_size())
                    selection_slice_map[features.id] = slice(selection_index, selection_index + selection_size)
                    index += size
                    selection_index += selection_size
                self.features_map[''] = features_id_map
                self.features_slice_map[''] = features_slice_map
                self.selection_slice_map[''] = selection_slice_map
            else:
                pfeatures = model_features
                for label in pfeatures.get_labels():
                    mfeatures = pfeatures.get_features(label)
                    self.labels.append(label)
                    mfeature_vector_slice = pfeatures.get_feature_vector_slice(label)
                    # set slices of merged feature set
                    self.feature_list_map[label] = pfeatures.get_features(label).get_feature_list()
                    self.features_list_map[label] = pfeatures.get_features(label).get_features_list()
                    selection_mask = None
                    if self.model.is_feature_selected():
                        selection_mask = self.model.get_selection_mask()[mfeature_vector_slice]
                    if selection_mask is None:
                        selection_mask = [True] * mfeatures.get_vector_size()
                    self.selection_mask_map[label] = selection_mask
                    self.vector_size += len(selection_mask)
                    self.selection_size += sum(selection_mask)
                    self.selection_ref_map[label] = np.zeros(sum(selection_mask), dtype=int)
                    self.feature_vector_slice_map[label] = mfeature_vector_slice
                    # make slices for each feature set in merged feature set
                    features_id_map = dict()
                    features_slice_map = dict()
                    selection_slice_map = dict()
                    index = 0
                    selection_index = 0
                    for features in mfeatures.get_features_list():
                        feature_vector_slice = mfeatures.get_feature_vector_slice(features)
                        features_id_map[features.id] = features
                        size = 0
                        selection_size = 0
                        for mask in selection_mask[feature_vector_slice]:
                            selection_size += 1 if mask else 0
                            size += 1
                        features_slice_map[features.id] = slice(index, index + features.get_vector_size())
                        selection_slice_map[features.id] = slice(selection_index, selection_index + selection_size)
                        index += size
                        selection_index += selection_size
                    self.features_map[label] = features_id_map
                    self.features_slice_map[label] = features_slice_map
                    self.selection_slice_map[label] = selection_slice_map
            self.labels = sorted(self.labels)

        def get_coef(self, vector_size):
            """Get coefficient vector of this model for a merged feature vector

            Args:
                vector_size (int): size of merged feature vector

            Returns:
                array, float: coefficient vector of this model, shift of regression value
            """
            coef = np.zeros(shape=(vector_size))
            local_coef = self.model.get_coef()
            local_shift = self.model.get_shift()
            label_index = 0
            for label in self.labels:
                offset = self.get_selection_offset(label)
                selection_ref = self.get_selection_ref(label)
                for index in range(len(selection_ref)):
                    coef[offset+selection_ref[index]] = local_coef[label_index+index]
                label_index += len(selection_ref)
            return coef, local_shift

        def get_local_data(self, data):
            """Retrieve a feature vector of this model from a merged feature vector

            Args:
                data: merged feature vectors

            Returns:
                local_data: feature vectors for this regression model
            """
            # retrieve data for this model
            if len(data.shape) == 2:
                local_data = np.zeros((data.shape[0], self.get_selection_size()))
                label_index = 0
                for label in self.labels:
                    offset = self.get_selection_offset(label)
                    selection_ref = self.get_selection_ref(label)
                    for index in range(len(selection_ref)):
                        local_data[:, label_index+index] = data[:, offset+selection_ref[index]]
                    label_index += len(selection_ref)
                return local_data
            else:
                local_data = np.zeros(self.get_selection_size())
                label_index = 0
                for label in self.labels:
                    offset = self.get_selection_offset(label)
                    selection_ref = self.get_selection_ref(label)
                    for index in range(len(selection_ref)):
                        local_data[label_index+index] = data[offset+selection_ref[index]]
                    label_index += len(selection_ref)
                return local_data

        def estimate_error(self, target_value, data, prediction_error=1.0):
            """Get estimates and square errors for given feature vectors

            Args:
                target_value (float): target value of the estimation
                data (matrix): feature vectors
                prediction_error (float, optional): ratio of square error for the error estimation. Defaults to 1.0.

            Returns:
                matrix, matrix: estimates and square errors
            """
            # estimate square error of estimation
            local_data = self.get_local_data(data)
            pred_error = self.model.get_prediction_std() * prediction_error
            if len(data.shape) == 2:
                estimate = self.model.predict_val(local_data)
                estimate_error = np.zeros(shape=estimate.shape)
                for index in range(len(estimate)):
                    est = estimate[index]
                    if est > target_value[1]:
                        estimate_error[index] = ((est - target_value[1]) / pred_error) ** 2
                    elif est < target_value[0]:
                        estimate_error[index] = ((est - target_value[0]) / pred_error) ** 2
            else:
                estimate = self.model.predict_single_val(local_data)
                estimate_error = 0
                if estimate > target_value[1]:
                    estimate_error = ((estimate - target_value[1]) / pred_error) ** 2
                elif estimate < target_value[0]:
                    estimate_error = ((estimate - target_value[0]) / pred_error) ** 2
            return estimate, estimate_error

    class TargetLabel(object):
        """Class for retrieve feature vector corresponding to label component for multi-target
        feature estimation

        Attributes:
            label (str): name of label component
            features_list (list): a list of feature sets
            feature_list (list): a list of features
            selection_mask (list): a selection mask of feature vector
            vector_slice (slice): a slice of a merged feature vector for the label component
            selection_slice (slice): a slice of a selected merged feature vector fo the label component
            max_atom (int): maximum number of atoms
            max_ring (int): maximum number of rings
            atom_range (dict): a map of atom symbol and min/max range
            ring_range (dict): a map of ring size and min/max range
            aring_range (dict): a map of aromatic ring size and min/max range
            atom_valence (dit): a map of atom symbol and valence
            checker (function): a structural constraint checker
            struct_index (list): a list of structural vector index
        """

        def __init__(self, label, moldata, features_list, feature_list, selection_mask, vector_slice, selection_slice):
            """constructor of FeatureEstimator.TargetLabel class

            Args:
                label (str): name of label component
                moldata (MolData): moldata object for the label
                features_list (list): a list of feature sets
                feature_list (list): a list of features
                selection_mask (list): a selection mask of feature vector
                vector_slice (slice): a slice of a merged feature vector for the label component
                selection_slice (slice): a slice of a selected merged feature vector for the label component
            """
            self.label = label
            self.moldata = moldata
            self.features_list = features_list
            self.feature_list = feature_list
            self.selection_mask = selection_mask
            self.vector_slice = vector_slice
            self.selection_slice = selection_slice
            self.max_atom = 0
            self.max_ring = 0
            self.atom_range = dict()
            self.ring_range = dict()
            self.aring_range = dict()
            self.atom_valence = dict()
            self.checker = None
            self.struct_index = []

        def get_label(self):
            """Get a name of label component

            Returns:
                str: a name of label component
            """
            return self.label

        def get_moldata(self):
            """Get a moldata object of label component

            Returns:
                MolData: a moldata object

            """
            return self.moldata

        def get_features_list(self):
            """Get a list of feature sets

            Returns:
                list: a list of features set
            """
            return self.features_list

        def get_feature_list(self):
            """Get a list of features

            Returns:
                list: a list of features
            """
            return self.feature_list

        def get_selection_mask(self):
            """Get a selection mask

            Returns:
                list: a selection mask
            """
            return self.selection_mask

        def get_max_atom(self):
            """Get max atom parameter

            Returns:
                int: max atom
            """
            return self.max_atom

        def get_max_ring(self):
            """Get max ring parameter

            Returns:
                int: max ring
            """
            return self.max_ring

        def get_atom_range(self):
            """Get a map of atom symbol and min/max range

            Returns:
                dict: map of min/max range
            """
            return self.atom_range

        def get_ring_range(self):
            """Get a map of ring size and min/max range

            Returns:
                dict: map of min/max range
            """
            return self.ring_range

        def get_aring_range(self):
            """Get a map of aromatic ring size and min/max range

            Returns:
                dict: map of min/max range
            """
            return self.aring_range

        def get_atom_valence(self):
            """Get a map of atom symbol and valence

            Returns:
                dict: map of atom valence
            """
            return self.atom_valence

        def get_vector_slice(self):
            """Get a slice of this feature vector in the merged feature vector

            Returns:
                slice: a slice of this feature vector
            """
            return self.vector_slice

        def get_selection_slice(self):
            """Get a slice of this feature vector in the selected merged feature vector

            Returns:
                slice: a slice of this feature vector
            """
            return self.selection_slice

        def get_structural_vector_index(self):
            """Get a list of structural vector index

            Returns:
                list: a list of indices
            """
            return self.struct_index

        def set_structural_vector_index(self, indices):
            """Set a list of structural vector index

            Args:
                indices (list): a list of indices
            """
            self.struct_index = indices

        def set_basic_ranges(self, atom_range, ring_range, aring_range, atom_valence):
            """Set min/max ranges of atom, ring, and aromatic ring

            Args:
                atom_range (dict): min/max range of atom symbol
                ring_range (dict): min/max range of ring size
                aring_range (dict): min/max range of aromatic ring size
                atom_valence (dict): max valence of atoms
            """
            self.atom_range = atom_range
            self.ring_range = ring_range
            self.aring_range = aring_range
            self.atom_valence = atom_valence

        def make_constraint_checker(self, max_atom, max_ring): 
            """make a structural constraint checker for label component

            Args:
                max_atom (int): maximum number of atoms for molecular generation
                max_ring (int): maximum number of rings for molecular generation
            """
            self.max_atom = max_atom
            self.max_ring = max_ring
            # make constraint checker on features
            feasible_fp = FeasibleFingerPrintVector(self.max_atom, self.max_ring)
            feasible_fp.fp_trajectory()
            self.checker = StructuralConstraint(self.label, self.feature_list, self.selection_mask,
                                                feasible_fp, self.max_atom, self.max_ring,
                                                self.atom_range, self.ring_range, self.aring_range)

        def constraint_check(self, data):
            """Check structural constraints for label component

            Args:
                data (np.ndarray): feature vectors

            Returns:
                array: constraint errors
            """
            if len(data.shape) == 2:
                const_violation = np.zeros(len(data))
                if self.checker is not None:
                    for index, feature_vector in enumerate(data):
                        iteration = 0
                        const_violation[index] = self.checker.check_constraint(feature_vector[self.selection_slice],
                                                                               iteration)
                return const_violation
            else:
                const_violation = 0
                if self.checker is not None:
                    iteration = 0
                    const_violation = self.checker.check_constraint(data[self.selection_slice], iteration)
                return const_violation

    def __init__(self, models):
        """Constructor of FeatureEvaluator class

        Args:
            models (list): a list of regression models
        """
        # check models
        target_properties = set()
        moldata = None
        for model in models:
            target_property = model.get_target_property()
            if target_property in target_properties:
                logger.error('models of duplicated target property %s', target_property)
                return None
            elif moldata is not None and moldata != model.get_moldata():
                logger.error('models of different moldata')
                return None
            else:
                target_properties.add(target_property)
                moldata = model.get_moldata()

        self.moldata = moldata
        self.target_models = dict()
        self.target_labels = dict()
        self.labels = []
        self.vector_size = 0
        self.features_list = []
        self.feature_list = []
        self.selection_mask = []
        self.data_mask = []
        self.feature_vector = None
        self.feature_dtype = None
        self.feature_range = None
        self.verbose = True

        # make TargetModel objects
        for model in models:
            self.target_models[model.get_target_property()] = self.TargetModel(model)

        # initialize target models and make target labels and the merged feature vector
        self.initialize()

    def initialize(self):
        """Initialize target models and target labels
        """
        # get union of feature set in each label
        features_id_map = defaultdict(list)
        features_map = defaultdict(dict)
        for target_property, target in self.target_models.items():
            for label in target.get_labels():
                if label not in self.labels:
                    self.labels.append(label)
                for features_id in target.get_features(label):
                    if features_id not in features_id_map[label]:
                        features_id_map[label].append(features_id)
                        features_map[label][features_id] = target.get_features(label)[features_id]
        self.labels = sorted(self.labels)
        for label in self.labels:
            features_id_list = features_id_map[label]
            features_id_map[label] = sorted(features_id_list)

        # make merged feature vector for each label
        self.vector_size = 0
        self.selection_size = 0
        self.features_list = []
        self.feature_list = []
        self.selection_mask = []
        for label in self.labels:
            features_id_list = features_id_map[label]
            label_vector_size = 0
            label_selection_size = 0
            label_features_list = []
            label_feature_list = []
            label_selection_mask = []
            # set label vector offset
            for target_property, target in self.target_models.items():
                target.set_selection_offset(label, self.selection_size)
            for features_id in features_id_list:
                features = features_map[label][features_id]
                label_features_list.append(features)
                label_feature_list.extend(features.get_feature_list())
                selection_mask = None
                slice_map = dict()
                for target_property, target in self.target_models.items():
                    if label in target.get_labels():
                        if features_id in target.get_features(label):
                            sl = target.get_features_slice(label)[features_id]
                            slice_map[target_property] = sl
                            mask = target.get_selection_mask(label)[sl]
                            if selection_mask is None:
                                selection_mask = mask
                            else:
                                selection_mask = [m1 or m2 for m1, m2 in zip(selection_mask, mask)]
                label_selection_mask.extend(selection_mask)
                # make an index reference for feature vector
                selection_pos = 0
                selection_ref_map = defaultdict(list)
                for index in range(len(selection_mask)):
                    if selection_mask[index]:
                        for target_property, sl in slice_map.items():
                            target = self.target_models[target_property]
                            mask = target.get_selection_mask(label)[sl]
                            if mask[index]:
                                selection_ref_map[target_property].append(label_selection_size+selection_pos)
                        selection_pos += 1
                for target_property, sl in slice_map.items():
                    # put vector ref to target objects
                    target = self.target_models[target_property]
                    selection_sl = target.get_selection_slice(label)[features_id]
                    selection_ref = selection_ref_map[target_property]
                    target.get_selection_ref(label)[selection_sl] = np.array(selection_ref, dtype=int)
                # update label vector_size
                label_vector_size += len(selection_mask)
                label_selection_size += sum(selection_mask)
            # make a target label
            vector_slice = slice(self.vector_size, self.vector_size + label_vector_size)
            selection_slice = slice(self.selection_size, self.selection_size + label_selection_size)
            self.target_labels[label] = self.TargetLabel(label,
                                                         self.moldata.get_subdata(label),
                                                         label_features_list,
                                                         label_feature_list,
                                                         label_selection_mask,
                                                         vector_slice,
                                                         selection_slice)
            self.features_list.extend(label_features_list)
            self.feature_list.extend(label_feature_list)
            self.selection_mask.extend(label_selection_mask)
            self.vector_size += label_vector_size
            self.selection_size += label_selection_size

        # get data type of feature vector
        feature_dtype = []
        index = 0
        for feature in self.feature_list:
            for idx in range(feature.get_vector_size()):
                feature_dtype.append(feature.get_dtype())
            index += feature.get_vector_size()
        self.feature_dtype = np.array(feature_dtype)

        # make mol data mask
        target_data_mask = None
        for target_property, target_model in self.target_models.items():
            target_df = self.moldata.get_property_vector()
            target_df[target_property] = target_df[target_property].astype(float)
            target = target_df[target_property].values
            target_data_mask = list(map(lambda x: not np.isnan(x), target))
            target_data_mask = update_data_mask(target_data_mask, self.moldata.get_mols_mask())
            for label in target_model.get_labels():
                subdata = self.moldata.get_subdata(label)
                target_data_mask = update_data_mask(target_data_mask, subdata.get_mols_mask())
                for features in target_model.get_features_list(label):
                    feature_mask = subdata.get_feature_mask(features.id)
                    target_data_mask = update_data_mask(target_data_mask, feature_mask)
        self.data_mask = target_data_mask

        # make a merged feature vector by joining feature vectors of labels
        df_index = [m.id for m in self.moldata.get_mols()]
        df = pd.DataFrame(index=df_index)
        fv_list = []
        for label in self.labels:
            target_label = self.target_labels[label]
            label_features_list = target_label.features_list
            merged_features = MergedFeatureSet(label_features_list)
            subdata = self.moldata.get_subdata(label)
            sub_fv = merged_features.make_feature_vector(subdata.get_mols())
            fv_list.append(sub_fv.rename(columns=lambda x: label + ':' + x))

        # get merged feature vector by masking invalid data and unselected features
        self.feature_vector = df.join(fv_list).values[target_data_mask]

    id_string_separator = '+++'
    """Separator of id string"""

    @classmethod
    def get_target_property_id_string(cls, models):
        """Make a target property id string from models

        Args:
            models (list): list of regression models
        """
        models = sorted(models, key=lambda x: x.get_target_property())
        rstr = ''
        for model in sorted(models, key=lambda x: x.get_target_property()):
            rstr += model.get_target_property() + cls.id_string_separator
        return rstr.rstrip(cls.id_string_separator)

    @classmethod
    def get_features_id_string(cls, models):
        """Make a features id string from models

        Args:
            models (list): list of regression models
        """
        models = sorted(models, key=lambda x: x.get_target_property())
        rstr = ''
        for model in sorted(models, key=lambda x: x.get_target_property()):
            rstr += model.get_features().get_id() + cls.id_string_separator
        return rstr.rstrip(cls.id_string_separator)

    @classmethod
    def get_models_id_string(cls, models):
        """Make a models id string from models

        Args:
            models (list): list of regression models
        """
        models = sorted(models, key=lambda x: x.get_target_property())
        rstr = ''
        for model in sorted(models, key=lambda x: x.get_target_property()):
            rstr += model.get_id() + cls.id_string_separator
        return rstr.rstrip(cls.id_string_separator)

    @classmethod
    def concat_id_string(cls, id_strings):
        """Concatenate id strings
        Args:
            id_strings (list): a list of strings of id
        """
        rstr = ''
        for id_string in id_strings:
            rstr += id_string + cls.id_string_separator
        return rstr.rstrip(cls.id_string_separator)

    @classmethod
    def split_id_string(cls, id_string):
        """Strip id string into individual models
        Args:
            id_string(str): string of id
        """
        return id_string.split(cls.id_string_separator)

    def is_linear_model(self):
        """Get if the models are linear regression

        Returns:
            bool: true if linear regression model
        """
        return all([tm.get_model().is_linear_model() for tm in self.target_models.values()])

    def has_single_target_label(self):
        """Get if it has only single target label

        Returns:
            bool: true if single target label
        """
        single_target = False
        for label, target in self.target_labels.items():
            label_slice = target.get_vector_slice()
            if self.get_vector_size() == label_slice.stop - label_slice.start:
                single_target = True
        return single_target

    def has_only_online_feature(self):
        """Get if it has only structural feature

        Returns:
            bool: true if only structural feature
        """
        all_online = True
        for label, target in self.target_labels.items():
            for features in target.get_features_list():
                if features.is_online_update():
                    continue
                else:
                    all_online = False
                    break
        return all_online

    def get_target_property_id(self):
        """Get target properties as id

        Returns:
            str: id of target property
        """
        models = [tm.get_model() for tm in self.target_models.values()]
        return self.get_target_property_id_string(models)

    def get_features_id(self):
        """Get merged features of models as id

        Returns:
            str: id of merged features
        """
        models = [tm.get_model() for tm in self.target_models.values()]
        return self.get_features_id_string(models)

    def get_models_id(self):
        """Get regression models as id

        Returns:
            str: id of regression models
        """
        models = [tm.get_model() for tm in self.target_models.values()]
        return self.get_models_id_string(models)

    def get_moldata(self):
        """Get MolData object

        Returns:
            MolData: MolData object
        """
        return self.moldata

    def get_label_feature_estimate(self, label):
        """Get feature estimation result stored in moldaata

        Args:
            label (str): label component name

        Returns:
            FeatureEstimationResult: feature estimate
        """
        return self.moldata.get_subdata(label).\
            get_feature_estimate(self.get_target_property_id(), self.get_features_id(), self.get_models_id())

    def get_target_properties(self):
        """Get a list of target properties sorted by their name

        Returns:
            list: a list of target properties
        """
        return sorted(list(self.target_models.keys()))

    def get_labels(self):
        """Get a list of label components sorted by their name

        Returns:
            list: a list of label components
        """
        return sorted(list(self.target_labels.keys()))

    def get_target_model(self, target_property):
        """Get a target model object

        Args:
            target_property (str): target property

        Returns:
            TargetModel: TargetModel object
        """
        if target_property in self.target_models:
            return self.target_models[target_property]
        else:
            return None

    def get_target_label(self, label):
        """Get a target label object

        Args:
            label (str): label component name

        Returns:
            TargetLabel: TargetLabel object
        """
        if label in self.target_labels:
            return self.target_labels[label]
        else:
            return None

    def get_feature_vector(self, with_mask=True):
        """Get a merged feature vector of moldata

        Args:
            with_mask (bool): flag to apply selection mask

        Returns:
            matrix: feature vectors
        """
        if with_mask:
            return self.feature_vector[:, self.selection_mask]
        else:
            return self.feature_vector

    def get_feature_dtype(self, with_mask=True):
        """Get a data type of feature vector value

        Args:
            with_mask (bool): flag to apply selection mask

        Returns:
            array: a data type of feature vector value
        """
        if with_mask:
            return self.feature_dtype[self.selection_mask]
        else:
            return self.feature_dtype

    def get_feature_range(self, with_mask=True):
        """Get ranges of feature vector elements

        Args:
            with_mask (bool): flag to apply selection mask

        Returns:
            matrix: feature ranges
        """
        if with_mask:
            return self.feature_range[self.selection_mask]
        else:
            return self.feature_range

    def get_vector_size(self):
        """Get a size of merged feature vector

        Returns:
            int: vector size
        """
        return self.vector_size

    def get_features_list(self):
        """Get a list of feature sets in the merged feature vector

        Returns:
            list: a list of feature sets
        """
        return self.features_list

    def get_feature_list(self):
        """Get a list of features in the merged feature vector

        Returns:
            list: a list of features
        """
        return self.feature_list

    def get_data_mask(self):
        """Get a valid data mask of the moldata

        Returns:
            list: a data mask
        """
        return self.data_mask

    def get_selection_mask(self):
        """Get a selection mask of the merged feature vector

        Returns:
            list: a selection mask
        """
        return self.selection_mask

    def get_label_features_list(self, label):
        """Get a list of feature sets for a feature vector of label component

        Args:
            label (str): label component name

        Returns:
            list: a list of feature sets
        """
        return self.get_target_label(label).get_features_list()

    def get_label_feature_list(self, label):
        """Get a list of features for a feature vector of label component

        Args:
            label (str): label component name

        Returns:
            list: a list of features
        """
        return self.get_target_label(label).get_feature_list()

    def get_label_selection_mask(self, label):
        """Get a selection mask for a feature vector of label component

        Args:
            label (str): label component name

        Returns:
            list: a selection mask
        """
        return self.get_target_label(label).get_selection_mask()

    def get_label_max_atom(self, label):
        """Get a max atom parameter of label component

        Args:
            label (str): label component name

        Returns:
            int: max atom
        """
        return self.get_target_label(label).get_max_atom()

    def get_label_max_ring(self, label):
        """Get a max ring parameter of label component

        Args:
            label (str): label component name

        Returns:
            int: max ring
        """
        return self.get_target_label(label).get_max_ring()

    def get_label_atom_range(self, label):
        """Get a min/max range of atoms of label component

        Args:
            label (str): label component name

        Returns:
            dict: min/max range of atoms
        """
        return self.get_target_label(label).get_atom_range()

    def get_label_ring_range(self, label):
        """Get a min/max range of ring sizes of label component

        Args:
            label (str): label component name

        Returns:
            dict: min/max range of ring sizes
        """
        return self.get_target_label(label).get_ring_range()

    def get_label_aring_range(self, label):
        """Get a min/max range of aromatic ring sizes of label component

        Args:
            label (str): label component name

        Returns:
            dict: min/max range of aromatic rings
        """
        return self.get_target_label(label).get_aring_range()

    def get_label_atom_valence(self, label):
        """Get an atom valence of label component

        Args:
            label (str): label component name

        Returns:
            dict: atom valence
        """
        return self.get_target_label(label).get_atom_valence()

    def prepare_root_struct_index(self):
        """Prepare indices of the root structure for the feature estimation and the structure generation.

        Returns:
            list: a list of feature vector indices for structure generation
        """
        whole_struct_index = []
        for label, target in self.target_labels.items():
            features_list = target.get_features_list()
            selection_mask = target.get_selection_mask()
            selection_slice = target.get_selection_slice()
            index = 0
            data_index = 0
            struct_index = []
            for features in features_list:
                for feature in features.get_feature_list():
                    for idx in range(feature.get_vector_size()):
                        if selection_mask[index + idx]:
                            if features.is_online_update():
                                struct_index.append(data_index)
                            data_index += 1
                    index += feature.get_vector_size()
            struct_index = sorted(struct_index)
            target.set_structural_vector_index(struct_index)
            whole_struct_index.extend([idx + selection_slice.start for idx in struct_index])
        return sorted(whole_struct_index)

    def prepare_evaluation(self, design_param, fix_condition, no_message=False):
        """ Define the range of feature vector values.

        Args:
            design_param (DesignParam): design parameter
            fix_condition (ComponentFixCondition): a fix condition
            no_message (bool): flag of print message

        Returns:
            array: a range of feature vector value
        """
        # extract individual range parameters
        range_params = design_param.get_range_params()
        max_atom = range_params['max_atom']
        max_ring = range_params['max_ring']
        sigma_ratio = range_params['sigma_ratio']
        count_tolerance = range_params['count_tolerance']
        count_min_ratio = range_params['count_min_ratio']
        count_max_ratio = range_params['count_max_ratio']

        # make feature range
        self.feature_range = self.calc_feature_range(self.feature_list, self.feature_vector,
                                                     sigma_ratio, count_tolerance,
                                                     count_min_ratio, count_max_ratio)

        # make constraint checker
        for label in self.labels:
            # get atom/ring range
            subdata = self.moldata.get_subdata(label)
            atom_range, ring_range, atom_range_map, ring_range_map, aring_range_map, atom_valence_map = \
                self.calc_basic_feature(subdata, self.data_mask,
                                        sigma_ratio, count_tolerance, count_min_ratio, count_max_ratio)
            if isinstance(max_atom, dict):
                if label in max_atom:
                    label_max_atom = int(max_atom[label])
                else:
                    label_max_atom = int(atom_range[1])
            else:
                if max_atom == 0:
                    label_max_atom = int(atom_range[1])
                else:
                    label_max_atom = max_atom
            if isinstance(max_ring, dict):
                if label in max_ring:
                    label_max_ring = int(max_ring[label])
                else:
                    label_max_ring = int(ring_range[1])
            else:
                if max_ring == 0:
                    label_max_ring = int(ring_range[1])
                else:
                    label_max_ring = max_ring

            # make constraint checker on features
            target_label = self.target_labels[label]
            target_label.set_basic_ranges(atom_range_map, ring_range_map, aring_range_map,
                                          atom_valence_map)
            target_label.make_constraint_checker(label_max_atom, label_max_ring)

            # get features for a label component
            label_feature_list = target_label.get_feature_list()
            label_selection_mask = target_label.get_selection_mask()
            label_selection_slice = target_label.get_selection_slice()

            # get if label component is fixed one
            label_fixed = label in fix_condition.get_labels()

            # overwrite feature ranges of fixed components if they are fixed
            if label_fixed:
                fix_vector = fix_condition.get_feature_vector(label)
                if fix_vector is not None:
                    select_index = 0
                    slice_index = 0
                    for index, mask in enumerate(self.selection_mask):
                        if mask:
                            if label_selection_slice.start <= select_index < label_selection_slice.stop:
                                self.feature_range[index] = fix_vector[slice_index]
                                slice_index += 1
                            select_index += 1
                else:
                    logger.error('cannot get feature vector for fixed molecules of {0}'.format(label))

            # print max atom and ring, vector size
            if not no_message:
                if label_fixed:
                    logger.info('[\'{0}\'] fixed component vector_size={1}'.
                          format(label, sum(label_selection_mask)))
                else:
                    logger.info('[\'{0}\'] max_atom={1} max_ring={2} vector_size={3}/{4}'.
                          format(label, label_max_atom, label_max_ring, sum(label_selection_mask),
                                 len(label_selection_mask)))

            if self.verbose:
                label_feature_dtype = self.feature_dtype[label_selection_slice]
                label_feature_range = self.feature_range[self.selection_mask][label_selection_slice]
                # print search range of feature vector
                if label_fixed:
                    range_str = '[\'{0}\'] fixed vector:'.format(label) + '{'
                else:
                    range_str = '[\'{0}\'] search range:'.format(label) + '{'
                index = 0
                data_index = 0
                for feature in label_feature_list:
                    headers = feature.get_header_list()
                    for idx in range(feature.get_vector_size()):
                        if label_selection_mask[index + idx]:
                            if label_feature_dtype[data_index] == int:
                                if label_fixed:
                                    range_str += '{0}:[{1:d}] '. \
                                        format(headers[idx],
                                               int(label_feature_range[data_index][0]))
                                else:
                                    range_str += '{0}:[{1:d},{2:d}] '. \
                                        format(headers[idx],
                                               int(label_feature_range[data_index][0]),
                                               int(label_feature_range[data_index][1]))
                            else:
                                if label_fixed:
                                    range_str += '{0}:[{1:.2f}] '. \
                                        format(headers[idx],
                                               label_feature_range[data_index][0])
                                else:
                                    range_str += '{0}:[{1:.2f},{2:.2f}] '. \
                                        format(headers[idx],
                                               label_feature_range[data_index][0],
                                               label_feature_range[data_index][1])
                            data_index += 1
                    index += feature.get_vector_size()
                range_str = range_str.strip() + '}'
                logger.info(range_str)

        all_index = 0
        select_index = 0
        for feature in self.feature_list:
            for index in range(feature.get_vector_size()):
                selected = self.selection_mask[all_index + index]
                if selected:
                    select_index += 1
            all_index += feature.get_vector_size()

        return self.feature_range

    def get_coef(self):
        """Get coefficient of regression models of target properties

        Returns:
            dict: map of a target property and coefficients of its regression model
        p"""
        coefs = dict()
        vector_size = sum(self.selection_mask)
        for target_property, target in self.target_models.items():
            local_coef, local_shift = target.get_coef(vector_size)
            local_std = target.get_model().get_prediction_std()
            coefs[target_property] = (local_coef, local_shift, local_std)
        return coefs

    def evaluate(self, target_values, data, prediction_error, fixed_labels):
        """Evaluate feature vector by square error of the estimation, and structural constraints

        Args:
            target_values (dict): a mapping of target property and target value
            data (matrix): feature vectors
            prediction_error (float): acceptable range of prediction error
            fixed_labels (list): list of fixed component labels

        Returns:
            array: total errors of feature vectors
        """
        total_error = np.zeros(len(data))
        for target_property, target in self.target_models.items():
            # estimate regression value
            target_value = target_values[target_property]
            estimate, estimate_error = target.estimate_error(target_value, data, prediction_error)
            for index in range(len(total_error)):
                total_error[index] = max(total_error[index], estimate_error[index])
        for target_label, target in self.target_labels.items():
            # evaluate constraint error
            if target_label not in fixed_labels:
                constraint_error = target.constraint_check(data)
                total_error += constraint_error
        return total_error

    def get_estimates_dataframe(self, label, target_values, molecules, fit_candidate, fix_condition):
        """Get dataframe of estimates and estimation errors for target properties

        Args:
            label (str): a component label
            target_values (dict): a mapping of target property and target value
            molecules (list): a list of molecules
            fit_candidate (dict): a mapping of molecule and matched candidate
            fix_condition (ComponentFixCondition): a fix condition

        Returns:
            DataFrame: estimates and estimation errors
        """
        df_index = [m.get_id() for m in molecules]
        df = pd.DataFrame(index=df_index)
        # make a merged feature vector for evaluation
        data = np.zeros((len(molecules), sum(self.selection_mask)))
        for index, molecule in enumerate(molecules):
            if molecule in fit_candidate:
                whole_vector = copy.copy(fit_candidate[molecule].get_whole_feature_vector())
            else:
                whole_vector = self.get_whole_feature_vector(label, molecule)
            data[index] = whole_vector
        # get estimates for target properties
        for target_property, target in self.target_models.items():
            target_value = target_values[target_property]
            if len(molecules) > 0:
                # estimate regression value
                estimate, estimate_error = target.estimate_error(target_value, data)
                df_estimate = pd.DataFrame(index=df_index, data=estimate,
                                           columns=['{0} {1}'.format(target_property, target_value)])
                df_error = pd.DataFrame(index=df_index, data=estimate_error,
                                        columns=['{0} score'.format(target_property)])
            else:
                df_estimate = pd.DataFrame(index=df_index,
                                           columns=['{0} {1}'.format(target_property, target_value)])
                df_error = pd.DataFrame(index=df_index,
                                        columns=['{0} score'.format(target_property)])
            df = df.join(df_estimate)
            df = df.join(df_error)
        return df

    def get_features_dataframe(self, label, molecules, fit_candidate):
        """Get dataframe of feature values in merged feature vector
        Args:
            label (str): a component label
            molecules (list): a list of molecules
            fit_candidate (dict): a mapping of molecule and matched candidate

        Returns:
            DataFrame: merged feature vector values
        """
        df_index = [m.get_id() for m in molecules]
        df = pd.DataFrame(index=df_index)
        # make a merged feature vector for the label component
        target_label = self.get_target_label(label)
        selection_mask = target_label.get_selection_mask()
        feature_vectors = []
        for index, molecule in enumerate(molecules):
            if molecule in fit_candidate:
                feature_vectors.append(self.get_merged_feature_vector(label, molecule, fit_candidate[molecule]))
            else:
                feature_vectors.append(self.get_merged_feature_vector(label, molecule))
        feature_vector = np.array(feature_vectors)
        feature_list = []
        for features in target_label.get_features_list():
            feature_list.extend(features.get_header_list())
        headers = [f for f, mask in zip(feature_list, selection_mask) if mask]
        if len(molecules) > 0:
            df_feature = pd.DataFrame(index=df_index, data=feature_vector, columns=headers)
        else:
            df_feature = pd.DataFrame(index=df_index, columns=headers)
        df = df.join(df_feature)
        return df

    def estimate_property(self, target_values, feature_vector):
        """Estimate target property values from a feature vector

        Args:
            target_values (dict): a mapping of target property and its value
            feature_vector (matrix): a feature vector

        Returns:
            dict: a dictionary of target property and estimated value
        """
        result = dict()
        for target_property, target in self.target_models.items():
            target_value = target_values[target_property]
            result[target_property] = target.estimate_error(target_value, feature_vector)
        return result

    def get_whole_feature_vector(self, label, molecule, candidate=None):
        """Make a whole merged feature vector from a molecule

        Args:
            label (str): component label
            molecule (SimpleMolecule): a molecule
            candidate (FeatureEstimationResult.Candidate, optional): feature estimate candidate. Defaults to None.

        Returns:
            array: whole feature vector
        """
        target_label = self.get_target_label(label)
        vector_slice = target_label.get_selection_slice()
        if candidate is None:
            # assume molecule is GeneratedMolecule
            whole_vector = molecule.get_vector_candidate().get_whole_feature_vector()
        else:
            whole_vector = candidate.get_whole_feature_vector()
        vector = np.array(whole_vector)
        for prop in self.get_target_properties():
            target_model = self.get_target_model(prop)
            if label in target_model.get_labels():
                features = target_model.get_model().get_features()
                feature_vector = features.make_feature_vector([molecule])
                selection_mask = target_model.get_selection_mask(label)
                selection_ref = target_model.get_selection_ref(label)
                selection_vector = feature_vector.values[0][selection_mask]
                for index, ref_index in enumerate(selection_ref):
                    vector[vector_slice][ref_index] = selection_vector[index]
        return vector

    def get_merged_feature_vector(self, label, molecule, candidate=None):
        """Make a merged feature vector of given label component from a molecule

        Args:
            label (str): component label
            molecule (list): a molecule
            candidate (FeatureEstimationResult.Candidate, optional): feature estimate candidate. Defaults to None.

        Returns:
            array: feature vector
        """
        target_label = self.get_target_label(label)
        vector_slice = target_label.get_selection_slice()
        if candidate is None:
            # assume molecule is GeneratedMolecule
            whole_vector = molecule.get_vector_candidate().get_whole_feature_vector()
        else:
            whole_vector = candidate.get_whole_feature_vector()
        vector = np.array(whole_vector[vector_slice])
        for prop in self.get_target_properties():
            target_model = self.get_target_model(prop)
            if label in target_model.get_labels():
                features = target_model.get_model().get_features()
                feature_vector = features.make_feature_vector([molecule])
                selection_mask = target_model.get_selection_mask(label)
                selection_ref = target_model.get_selection_ref(label)
                selection_vector = feature_vector.values[0][selection_mask]
                for index, ref_index in enumerate(selection_ref):
                    vector[ref_index] = selection_vector[index]
        return vector

    def get_merged_feature_vector_range(self, label, molecule, candidate=None):
        """Make a merged feature vector range of given label component from a molecule

        Args:
            label (str): component label
            molecule (SimpleMolecule): a molecule
            candidate (FeatureEstimationResult.Candidate, optional): feature estimate candidate. Defaults to None.

        Returns:
            array: feature vector range
        """
        vector = self.get_merged_feature_vector(label, molecule, candidate=candidate)
        return np.array([vector, vector]).T

    def detail_info(self, target_values, candidate, raw_candidate, fixed_labels):
        """Print string of details of evaluation of a feature vector

        Args:
            target_values (dict): a mapping of target property and target value
            candidate (array): a feature vector
            raw_candidate (array): a feature vector before rounding
            fixed_labels (list): a list of fixed component labels

        Returns:
            str: print string
        """
        rstr = ''
        for target_property, target in self.target_models.items():
            # estimate regression value
            target_value = target_values[target_property]
            estimate, estimate_error = target.estimate_error(target_value, candidate)
            raw_estimate, raw_estimate_error = target.estimate_error(target_value, raw_candidate)
            rstr += '  property:\'{0}\' target={1} ({2:.3f}) estimate={3:.3f} ({4:.3f}) error={5:.3f}\n'.\
                format(target_property,
                       target_values[target_property], target.get_model().get_prediction_std(),
                       estimate, raw_estimate, estimate_error)
        for target_label, target in self.target_labels.items():
            # evaluate constraint error
            if target_label not in fixed_labels:
                constraint_error = target.constraint_check(candidate)
                rstr += '  component:\'{0}\' constraint violation={1}\n'.\
                    format(target_label, constraint_error)
        return rstr.rstrip('\n')

    @staticmethod
    def calc_basic_feature(moldata, data_mask, sigma_ratio, count_tolerance, count_min_ratio, count_max_ratio):
        """Get ranges of the number of atoms and rings from data

        Args:
            moldata (MolData): molecule data
            data_mask (list): valid data mask
            sigma_ratio (float): std multiplier for search range.
            count_tolerance (int): tolerance of counting feature for search range.
            count_min_ratio (float): min value multiplier for search range.
            count_max_ratio (float): max value multiplier for search range.

        Returns:
            nparray, nparray, dict, dict, dict: min/max ranges of atoms and rings, and a map of atom, ring,
                aring and min/max ranges
        """
        fs_atom, fs_atom_mask = HeavyAtomExtractor(moldata).extract()
        fs_num_atom, fs_num_atom_mask = FeatureSumOperator(moldata, fs_atom).extract()
        fs_ring, fs_ring_mask = RingExtractor(moldata).extract()
        fs_num_ring, fs_num_ring_mask = FeatureSumOperator(moldata, fs_ring).extract()
        fs_aring, fs_aring_mask = AromaticRingExtractor(moldata).extract()
        mfs_basic = MergedFeatureSet([fs_num_atom, fs_num_ring, fs_atom, fs_ring, fs_aring])
        mfs_basic_mask = data_mask
        fv_basic = mfs_basic.make_feature_vector(moldata.get_mols()).values[mfs_basic_mask]
        value_min = np.zeros(fv_basic.shape[1])
        value_max = np.zeros(fv_basic.shape[1])
        data_mean = np.mean(fv_basic, axis=0)
        data_std = np.std(fv_basic, axis=0)
        if count_min_ratio is not None:
            data_min = np.min(fv_basic, axis=0)
        if count_max_ratio is not None:
            data_max = np.max(fv_basic, axis=0)
        for index, f in enumerate(mfs_basic.get_feature_list()):
            dmean = data_mean[index]
            dstd = data_std[index]
            value_min[index] = dmean - sigma_ratio * dstd - count_tolerance
            value_max[index] = dmean + sigma_ratio * dstd + count_tolerance
            if count_min_ratio is not None:
                dmin = data_min[index]
                value_min[index] = min(value_min[index],
                                       count_min_ratio * dmin - count_tolerance)
            if count_max_ratio is not None:
                dmax = data_max[index]
                value_max[index] = max(value_max[index],
                                       count_max_ratio * dmax + count_tolerance)
            value_min[index] = int(max(0, math.floor(value_min[index])))
            value_max[index] = int(math.ceil(value_max[index]))
        feature_range = np.column_stack((value_min, value_max))

        # extract range of each type
        atom_range = feature_range[0]
        ring_range = feature_range[1]
        atom_range_map = dict()
        ring_range_map = dict()
        aring_range_map = dict()
        atom_valence_map = dict()
        for index, f in enumerate(mfs_basic.get_feature_list()):
            if isinstance(f, HeavyAtomExtractor.Feature):
                atom_range_map[f.get_symbol()] = feature_range[index].astype(int)
                atom_valence_map[f.get_symbol()] = f.get_valence()
            elif isinstance(f, RingExtractor.Feature):
                ring_range_map[f.get_ring_size()] = feature_range[index].astype(int)
            elif isinstance(f, AromaticRingExtractor.Feature):
                aring_range_map[f.get_ring_size()] = feature_range[index].astype(int)

        return atom_range, ring_range, atom_range_map, ring_range_map, aring_range_map, atom_valence_map

    @staticmethod
    def calc_feature_range(feature_list, feature_vector,
                           sigma_ratio, count_tolerance, count_min_ratio, count_max_ratio):
        """Get ranges of feasible feature values from actual data.

        For cardinal number,
        min range is minimum of mean - sigma_ratio * std,
        min_value - count_tolerance, and count_min_ratio * min_value.
        max range is maximum of mean + sigma_ratio * std,
        max_value + count_tolerance, and count_max_ratio * max_value.

        For other number,
        min range is minimum of mean - sigma_ratio * std, and min_value.
        max range is maximum of mean + sigma_ratio * std, and max_value.

        Args:
            feature_list (list): list of features
            feature_vector (array): feature value vector
            sigma_ratio (float): std multiplier for search range.
            count_tolerance (int): tolerance of counting feature for search range.
            count_min_ratio (float): min value multiplier for search range.
            count_max_ratio (float): max value multiplier for search range.

        Returns:
            array: an array of min/max values of features
         """
        # set valid range of features
        value_min = np.zeros(feature_vector.shape[1], dtype=np.float)
        value_max = np.zeros(feature_vector.shape[1], dtype=np.float)
        data_mean = np.mean(feature_vector, axis=0)
        data_std = np.std(feature_vector, axis=0)
        data_min = np.min(feature_vector, axis=0)
        data_max = np.max(feature_vector, axis=0)
        index = 0
        for feature in feature_list:
            headers = feature.get_header_list()
            for idx in range(feature.get_vector_size()):
                dmean = data_mean[index+idx]
                dstd = data_std[index+idx]
                dmin = data_min[index+idx]
                dmax = data_max[index+idx]
                if logger.isEnabledFor(logging.INFO):
                    logger.info('data range:%s mean=%f std=%f [%f, %f]',
                                headers[idx], dmean, dstd, dmin, dmax)
                value_min[index+idx] = dmean - sigma_ratio * dstd - count_tolerance
                value_max[index+idx] = dmean + sigma_ratio * dstd + count_tolerance
                if count_min_ratio is not None:
                    value_min[index+idx] = min(value_min[index+idx],
                                               count_min_ratio * dmin - count_tolerance)
                if count_max_ratio is not None:
                    value_max[index+idx] = max(value_max[index+idx],
                                               count_max_ratio * dmax + count_tolerance)
                if feature.get_dtype() == int:
                    (domain_min, domain_max) = feature.get_domain()
                    value_min[index+idx] = int(math.floor(value_min[index+idx]))
                    if domain_min is not None:
                        value_min[index+idx] = max(domain_min, value_min[index+idx])
                    value_max[index+idx] = int(math.ceil(value_max[index+idx]))
                    if domain_max is not None:
                        value_max[index+idx] = min(domain_max, value_max[index+idx])
            index += feature.get_vector_size()
        feature_range = np.column_stack((value_min, value_max))
        return feature_range


# -----------------------------------------------------------------------------
# StructuralConstraint: checking structural constraint of feature vector
# -----------------------------------------------------------------------------

class StructuralConstraint(object):
    """A class for checking structural constraint of given sub-structures in a feature vector.

    Attributes:
        label (str): component label
        max_atom (int): maximum number of atoms to be used in structure generation
        max_ring (int): maximum number of rings to be used in structure generation
        feature_fp (FeasibleFingerPrintVector): feasibility checker of finger print structure
        feature_index (dict): index of a feature vector for a i-th feature
        atom_feature (list): a list of heavy atom features selected in a feature vector
        all_atom_feature (list): a list of all the heavy atom features
        all_atom_set (set): a set of all the heavy atom features
        ring_feature (list): a list of ring features selected in a feature vector
        all_ring_feature (list): a list of all the ring features
        aring_feature (list): a list of aromatic ring features selected in a feature vector
        all_aring_feature (list): a list of all the aromatic ring features
        fp_feature (list): a list of finger print features selected in a feature vector
        all_fp_feature (list): a list of all the finger print features
        fp_root_atom (dict): a mapping of finger print features and a root atom
        fp_feature_atom (dict): a mapping of finger print features and a atom counter
    """

    def __init__(self, label, feature_list, selection_mask, feature_fp,
                 max_atom, max_ring, atom_range, ring_range, aring_range): 
        """Get penalty function closure as constrains on feasible feature values.

        Args:
            label (str): component label
            feature_list (list): a list of features
            selection_mask (list): a list of flag of feature value selection
            feature_fp (FeasibleFingerPrintVector): feasibility checker of fingerprint
            max_atom (int): maximum number of heavy atoms.
            max_ring (int): maximum number of rings.
            atom_range (dict): range of the number of atoms
            ring_range (dict): range of the number of rings
            aring_range (dict): range of the number fo aromatic rings

        Returns:
            function object: function closure for penalty calculation
        """
        self.label = label
        self.max_atom = max_atom
        self.max_ring = max_ring
        self.atom_range = atom_range
        self.ring_range = ring_range
        self.aring_range = aring_range
        self.feature_fp = feature_fp
        self.feature_index = dict()
        self.atom_feature = []
        self.all_atom_feature = []
        self.all_atom_set = set()
        self.ring_feature = []
        self.all_ring_feature = []
        self.aring_feature = []
        self.all_aring_feature = []
        self.fp_feature = []
        self.all_fp_feature = []
        self.fp_root_atom = dict()
        self.fp_feature_atom = dict()

        # classify all feature types
        index = 0
        data_index = 0
        for feature in feature_list:
            self.feature_index[feature] = data_index
            if isinstance(feature, HeavyAtomExtractor.Feature):
                self.all_atom_feature.append(feature)
                self.all_atom_set.add(feature.get_symbol())
                if selection_mask[index]:
                    self.atom_feature.append(feature)
            elif isinstance(feature, RingExtractor.Feature):
                self.all_ring_feature.append(feature)
                if selection_mask[index]:
                    self.ring_feature.append(feature)
            elif isinstance(feature, AromaticRingExtractor.Feature):
                self.all_aring_feature.append(feature)
                if selection_mask[index]:
                    self.aring_feature.append(feature)
            elif isinstance(feature, FingerPrintStructureExtractor.Feature):
                self.all_fp_feature.append(feature)
                self.fp_feature_atom[feature] = feature.get_graph().get_atom_count()
                for v in feature.get_graph().vertices:
                    if v.root == 1:
                        self.fp_root_atom[feature] = v
                        break
                if selection_mask[index]:
                    self.fp_feature.append(feature)
            for idx in range(feature.get_vector_size()):
                if selection_mask[index + idx]:
                    data_index += 1
            index += feature.get_vector_size()

        # get available atoms
        available_atom = set()
        for atom in self.atom_range:
            available_atom.add(atom)

        for feature, atom_count in self.fp_feature_atom.items():
            for atom, count in atom_count.items():
                if atom not in available_atom:
                    logger.error('{0}: atom {1} of {2} not in atom_range'.format(label, atom, feature.id))

    def check_constraint(self, feature_vector, iteration):
        """Function closure for penalty calculation.

        Args:
            feature_vector (array): feature vector.
            iteration (int): number of iteration

        Returns:
            float: penalty value representing feasibility of feature vector
        """
        penalty = 0.0

        if len(feature_vector) == 0:
            return penalty

        max_atom = self.max_atom
        max_ring = self.max_ring

        atom_count = Counter()
        for atom_f in self.atom_feature:
            atom = atom_f.get_symbol()
            atom_count[atom] = feature_vector[self.feature_index[atom_f]]
        total_atom = sum(atom_count.values())
        ring_count = Counter()
        for ring_f in self.ring_feature:
            ring_count[ring_f.get_ring_size()] = feature_vector[self.feature_index[ring_f]]
        total_ring = sum(ring_count.values())
        aring_count = Counter()
        for aring_f in self.aring_feature:
            aring_count[aring_f.get_ring_size()] = feature_vector[self.feature_index[aring_f]]
        total_aring = sum(aring_count.values())

        if len(self.atom_feature) > 0 and max_atom > 0:
            # constraint of number of atom
            # total atom <= max_atom
            penalty0 = abs(min(max_atom - total_atom, 0))
            penalty += penalty0
            if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug('penalty:%d num_atom max=%d total=%d',
                             penalty0, max_atom, total_atom)

        ring_atom = 0
        if len(self.atom_feature) > 0 and len(self.ring_feature) + len(self.aring_feature) > 0:
            # constraint ring to atom
            # total atom >= ring_atom
            # aromatic rings share only one edge
            for ring_size, count in aring_count.items():
                ring_atom += (ring_size - 3) * count
            if total_aring > 1:
                ring_atom += 4
            elif total_aring > 0:
                ring_atom += 3
            # rings share (atom_size-1) at most
            ring_atom += (total_ring - total_aring) / 2
            max_ring_atom = 0
            for ring_size, count in ring_count.items():
                if count > aring_count[ring_size]:
                    if total_aring > 0:
                        max_ring_atom = max(max_ring_atom, ring_size - 3)
                    else:
                        max_ring_atom = max(max_ring_atom, ring_size - 1)
            ring_atom += max_ring_atom
            if len(self.atom_feature) == len(self.all_atom_feature):
                penalty0 = abs(min(total_atom - ring_atom, 0))
                penalty += penalty0
                if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('penalty:%d ring_atom total=%d ring=%d',
                                 penalty0, total_atom, ring_atom)
            elif max_atom > 0:
                penalty0 = abs(min(max_atom - ring_atom, 0))
                penalty += penalty0
                if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('penalty:%d ring_atom max=%d ring=%d',
                                 penalty0, max_atom, ring_atom)

        if len(self.ring_feature) > 0 and len(self.aring_feature) > 0:
            # constraint aring to ring
            # total ring_size >= aring_size
            for ring_size, count in ring_count.items():
                penalty0 = abs(min(count - aring_count[ring_size], 0))
                penalty += penalty0
                if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('penalty:%d aring-ring size=%d ring=%d aring=%d',
                                 penalty0, ring_size, count, aring_count[ring_size])

        if len(self.fp_feature) > 0:
            if 0 < len(self.aring_feature) == len(self.all_aring_feature):
                # constraint sub-structure to aromatic ring
                for sub_f in self.fp_feature:
                    if sub_f.get_graph().num_aromatic_atom() > 0 and \
                            feature_vector[self.feature_index[sub_f]] > 0:
                        penalty0 = abs(min(total_aring - 1, 0))
                        penalty += penalty0
                        if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('penalty:%d sub-aring sub=%s aring=%d',
                                         penalty0, sub_f.id, total_aring)

            if 0 < len(self.ring_feature) == len(self.all_ring_feature):
                # constraint sub-structure to ring
                for sub_f in self.fp_feature:
                    if sub_f.get_graph().num_ring_atom() > 0 and \
                            feature_vector[self.feature_index[sub_f]] > 0:
                        penalty0 = abs(min(total_ring - 1, 0))
                        penalty += penalty0
                        if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('penalty:%d sub-ring sub=%s ring=%d',
                                         penalty0, sub_f.id, total_ring)

        if len(self.fp_feature) > 0:
            # constraint fp_feature to atom
            # atom > atom in fp_structure
            # constraint by root atom
            fp_root_atom_count = Counter()
            for fp_f in self.fp_feature:
                fp_root_atom_count[fp_f.get_root_atom()] += feature_vector[self.feature_index[fp_f]]
            for atom, count in fp_root_atom_count.items():
                if atom in atom_count:
                    max_count = atom_count[atom]
                else:
                    max_count = self.atom_range[atom][1]
                penalty0 = abs(min(max_count - count, 0))
                penalty += penalty0
                if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('penalty:%d fp-root-atom atom=%s atom=%d fp_atom=%d',
                                 penalty0, atom, max_count, count)
            # constraint by all atom
            fp_other_atom_count = Counter()
            for fp_f in self.fp_feature:
                if feature_vector[self.feature_index[fp_f]] > 0:
                    for atom, count in self.fp_feature_atom[fp_f].items():
                        fp_other_atom_count[atom] = max(fp_other_atom_count[atom], count)
                        if atom in atom_count:
                            max_count = atom_count[atom]
                        elif atom in self.atom_range:
                            max_count = self.atom_range[atom][1]
                        penalty0 = abs(min(max_count - count, 0))
                        penalty += penalty0
                        if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                            logger.debug('penalty:%d fp-allatom fp=%s atom=%s atom=%d fp_atom=%d',
                                         penalty0, fp_f.id, atom, max_count, count)
            fp_total_atom = 0
            for atom, count in fp_other_atom_count.items():
                fp_total_atom += max(max(count, fp_root_atom_count[atom]), atom_count[atom])
            for atom, count in atom_count.items():
                if atom not in fp_other_atom_count:
                    fp_total_atom += count
            penalty0 = abs(min(max_atom - fp_total_atom, 0))
            penalty += penalty0
            if penalty0 > 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug('penalty:%d fp_all_total_atom=%d max_atom=%d',
                             penalty0, fp_total_atom, max_atom)

        if penalty > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('total penalty:%d', penalty + 1)
            return penalty + 1

        if len(self.fp_feature) > 0:
            num_atom_f = len(self.all_atom_feature)
            num_ring_f = len(self.all_ring_feature)
            # get atoms
            atom = [0, 0, 0, 0]
            atom_f_count = 0
            for atom_f in self.atom_feature:
                fv_value = int(feature_vector[self.feature_index[atom_f]])
                atom[min(4, atom_f.get_valence()) - 1] += fv_value
                atom_f_count += 1
            # get valence of unspecified atom
            anonymous_valence = 0
            for atom_f in self.all_atom_feature:
                if atom_f not in self.atom_feature:
                    anonymous_valence = max(anonymous_valence,
                                            min(4, atom_f.get_valence()))
            # define atom comparison operation
            complete_atom = (atom_f_count == num_atom_f or total_atom == max_atom)
            if complete_atom:
                atom_op = 'E'
            else:
                atom_op = 'G'
            # get rings
            ring = 0
            ring_f_count = 0
            for ring_f in self.ring_feature:
                fv_value = int(feature_vector[self.feature_index[ring_f]])
                ring += fv_value
                ring_f_count += 1
            # define ring comparison operation
            if complete_atom and (ring_f_count == num_ring_f or total_ring == max_ring):
                ring_op = 'E'
            else:
                ring_op = 'G'
            if 0 <= ring <= 1:
                ring_action = ring
            else:
                ring_action = int(math.floor(ring / 2)) + 1
            # get fp
            fp = [0, 0, 0, 0]
            atom_bond = Counter()
            for fp_f in self.fp_feature:
                root_atom = self.fp_root_atom[fp_f]
                fv_value = feature_vector[self.feature_index[fp_f]]
                if root_atom.num_edge() <= 4:
                    fp[root_atom.num_edge() - 1] += int(fv_value)
                if fv_value > 0:
                    valence = min(4, root_atom.num_valence())
                    real_valence = root_atom.num_edge()
                    if real_valence < valence:
                        # update atom counter
                        atom[valence - 1] -= int(fv_value)
                        atom[real_valence - 1] += int(fv_value)
                    for e in root_atom.edges:
                        bond_order = e.get_bond_order()
                        if bond_order >= 2.0:
                            if root_atom.num_valence() <= 4:
                                atom_bond[(root_atom.atom, root_atom.num_valence(), bond_order)] -= 1
                            if e.end.num_valence() <= 4:
                                atom_bond[(e.end.atom, e.end.num_valence(), bond_order)] += 1
            for (a, valence, bond_order), count in atom_bond.items():
                if count > 0:
                    # we cannot use aromatic bond as a constraint because of
                    # many exceptional connection
                    # if bond == 1.5:
                    #    atom[valence-1] -= int(count/2)
                    #    atom[valence-1-1] += int(count/2)
                    if bond_order == 2.0:
                        atom[valence - 1] -= int(count)
                        atom[valence - 1 - 1] += int(count)
                    elif bond_order == 3.0:
                        atom[valence - 1] -= int(count)
                        atom[valence - 2 - 1] += int(count)

            # define fp comparison operation
            complete_fp = (sum(atom_count.values()) == sum(fp))
            if complete_atom and complete_fp:
                fp_op = 'E'
            else:
                fp_op = 'G'

            # fix minus atom count
            for index in range(len(atom)):
                atom[index] = max(atom[index], 0)

            # check feasibility
            if not self.feature_fp.check_feasibility(atom, ring_action, fp, atom_op, ring_op, fp_op,
                                                     anonymous_valence, ring_atom):
                penalty += 1.0

        return penalty


class FeasibleFingerPrintVector:
    """Check feasibility of fingerprint structure vector.

    Attributes:
        total_atom (int): maximum number of atoms
        total_ring (int): maximum number of rings
        valence (int): maximum number of valence of molecular structure
        trajectory (dict): sets of feasible fingerprint structure vectors partitioned by number of atoms
        feasible_set (set): cache of feasible points
        infeasible_set (set): cache of infeasible points
    """

    def __init__(self, total_atom, total_ring, valence=4):
        """Constructor of FeasibleFingerPrintVector.

        Args:
            total_atom (int): maximum number of atoms
            total_ring (int): maximum number of rings
            valence (int, optional): maximum number of valence of molecular structure. Defaults to 4.
        """
        self.total_atom = total_atom
        self.total_ring = total_ring
        self.valence = valence
        self.trajectory = defaultdict(self.dset)
        self.feasible_set = set()
        self.infeasible_set = set()

    @staticmethod
    def dset():
        return defaultdict(set)

    def check_feasibility(self, atom, ring, fp, atom_op, ring_op, fp_op,
                          anonymous_valence, ring_atom):
        """Check structural feasibility of given feature vector.

        Args:
            atom (list): number of atoms for each valence (1~4)
            ring (int): number of rings
            fp (list): a list of number of 1/2/3/4 bond fingerprint
            atom_op (str): check operation for atom ('E' or 'G')
            ring_op (str): check operation for ring ('E' or 'G')
            fp_op (str): check operation for bond fingerprint ('E' or 'G')
            anonymous_valence (int): valence of atoms not counted in atom
            ring_atom: the number of atoms in rings

        Returns:
             bool: true if feasible
        """
        # make an input string for caching
        fp_str = '([{0},{1},{2},{3}],{4:.0f},{5:.0f},{6:.0f},{7:.0f},{8:.0f})'. \
            format(atom[0], atom[1], atom[2], atom[3], ring, fp[0], fp[1], fp[2], fp[3])
        op_str = '({0},{1},{2})'. \
            format(atom_op, ring_op, fp_op)
        sol_str = '{0} {1}'.format(fp_str, op_str)

        # check cached result
        if sol_str in self.feasible_set:
            return True
        if sol_str in self.infeasible_set:
            return False

        for num_atom in sorted(self.trajectory.keys()):
            if num_atom > self.total_atom:
                break
            # filter by atom
            if atom_op == 'E':
                if num_atom < atom[0] + atom[1] + atom[2] + atom[3]:
                    continue
                elif num_atom > atom[0] + atom[1] + atom[2] + atom[3]:
                    break
            elif atom_op == 'G':
                if num_atom < atom[0] + atom[1] + atom[2] + atom[3]:
                    continue
            for num_ring in sorted(self.trajectory[num_atom].keys()):
                if 0 < num_ring > self.total_ring:
                    break
                # filter by ring
                if ring_op == 'E':
                    if num_ring < ring:
                        continue
                    elif num_ring > ring:
                        break
                elif ring_op == 'G':
                    if num_ring < ring:
                        continue
                # check feasibility
                for v in self.trajectory[num_atom][num_ring]:
                    # fileter by ring atom (2-hand at least)
                    if v[1]+v[2]+v[3] < ring_atom:
                        continue
                    # filter by fp1
                    if fp_op == 'E':
                        if v[0] != fp[0]:
                            continue
                    elif fp_op == 'G':
                        if v[0] < fp[0]:
                            continue
                        if v[0] < atom[0]:
                            continue
                    # filter by fp2
                    if fp_op == 'E':
                        if v[1] != fp[1]:
                            continue
                    elif fp_op == 'G':
                        if v[1] < fp[1]:
                            continue
                        if v[0]+v[1] < atom[0]+atom[1]:
                            continue
                        if 0 < anonymous_valence < 2 and v[1]+v[2]+v[3] > atom[1]+atom[2]+atom[3]:
                            continue
                    # filter by fp3
                    if fp_op == 'E':
                        if v[2] != fp[2]:
                            continue
                    elif fp_op == 'G':
                        if v[2] < fp[2]:
                            continue
                        if v[0]+v[1]+v[2] < atom[0]+atom[1]+atom[2]:
                            continue
                        if 0 < anonymous_valence < 3 and v[2]+v[3] > atom[2]+atom[3]:
                            continue
                    # filter by fp4
                    if fp_op == 'E':
                        if v[3] != fp[3]:
                            continue
                    elif fp_op == 'G':
                        if v[3] < fp[3]:
                            continue
                        if v[0]+v[1]+v[2]+v[3] < atom[0]+atom[1]+atom[2]+atom[3]:
                            continue
                        if 0 < anonymous_valence < 4 and v[3] > atom[3]:
                            continue
                    # there is a feasible vector
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('feasible: %s', sol_str)
                    # for debugging
                    # print('feasible: {0}'.format(sol_str))
                    self.feasible_set.add(sol_str)
                    return True
        # there is no feasible vector
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('infeasible: %s', sol_str)
        # for debugging
        # print('infeasible: {0}'.format(sol_str))
        self.infeasible_set.add(sol_str)
        return False

    def fp_trajectory(self):
        """Enumeration all the feasible fingerprint structure vectors.
        """
        # add initial trajectory
        fp = (2, 0, 0, 0)
        self.trajectory[self.fp_atom_value(fp)][0].add(fp)
        for num_atom in range(max(self.trajectory.keys()), self.total_atom+1):
            if max(self.trajectory.keys()) < num_atom:
                break
            for num_ring in range(0, max(self.total_atom, self.total_ring)):
                if max(self.trajectory[num_atom].keys()) < num_ring:
                    break
                for fp in self.trajectory[num_atom][num_ring]:
                    # add atom operation
                    if num_atom < self.total_atom:
                        if fp[0] >= 1 and self.valence >= 2:
                            new_fp = (fp[0], fp[1]+1, fp[2], fp[3])
                            self.trajectory[num_atom+1][num_ring].add(new_fp)
                        if fp[1] >= 1 and self.valence >= 3:
                            new_fp = (fp[0]+1, fp[1]-1, fp[2]+1, fp[3])
                            self.trajectory[num_atom+1][num_ring].add(new_fp)
                        if fp[2] >= 1 and self.valence >= 4:
                            new_fp = (fp[0]+1, fp[1], fp[2]-1, fp[3]+1)
                            self.trajectory[num_atom+1][num_ring].add(new_fp)

                    # add bond operation
                    fp_sum0 = fp[0]+fp[1]+fp[2]+fp[3]
                    fp_sum1 = fp[1]+fp[2]+fp[3]
                    if fp[0] >= 2 and fp_sum0 >= 3 and fp_sum1 >= 1 and self.valence >= 2:
                        new_fp = (fp[0]-2, fp[1]+2, fp[2], fp[3])
                        self.trajectory[num_atom][num_ring+1].add(new_fp)
                    if fp[0] >= 1 and fp[1] >= 1 and fp_sum0 >= 4 and fp_sum1 >= 2 and self.valence >= 3:
                        new_fp = (fp[0]-1, fp[1], fp[2]+1, fp[3])
                        self.trajectory[num_atom][num_ring+1].add(new_fp)
                    if fp[1] >= 2 and fp_sum1 >= 4 and self.valence >= 3:
                        new_fp = (fp[0], fp[1]-2, fp[2]+2, fp[3])
                        self.trajectory[num_atom][num_ring+1].add(new_fp)
                    if fp[0] >= 1 and fp[2] >= 1 and fp_sum0 >= 5 and fp_sum1 >= 2 and self.valence >= 4:
                        new_fp = (fp[0]-1, fp[1]+1, fp[2]-1, fp[3]+1)
                        self.trajectory[num_atom][num_ring+1].add(new_fp)
                    if fp[1] >= 1 and fp[2] >= 1 and fp_sum0 >= 5 and fp_sum1 >= 3 and self.valence >= 4:
                        new_fp = (fp[0], fp[1]-1, fp[2], fp[3]+1)
                        self.trajectory[num_atom][num_ring+1].add(new_fp)
                    if fp[2] >= 2 and fp_sum1 >= 5 and self.valence >= 4:
                        new_fp = (fp[0], fp[1], fp[2]-2, fp[3]+2)
                        self.trajectory[num_atom][num_ring+1].add(new_fp)
        # add special trajectory
        self.trajectory[0][0].add((0, 0, 0, 0))
        self.trajectory[1][0].add((1, 0, 0, 0))

    @staticmethod
    def fp_atom_value(fp):
        """Get atom number from fingerprint structure vector.

        Returns:
            int: the number of atoms
        """
        return fp[0]+fp[1]+fp[2]+fp[3]

    @staticmethod
    def fp_bond1_value(fp):
        """Get the number of 1 bond sub-structures from fingerprint structure vector.

        Returns:
            int: the number of 1 bond sub-structures
        """
        return int((1*fp[0]+2*fp[1]+3*fp[2]+4*fp[3])/2)

    @staticmethod
    def fp_bond2_value(fp):
        """Get the number of 2 bond sub-structures from fingerprint structure vector.

        Returns:
            int: the number of 2 bond sub-structures
        """
        return 1*fp[1]+3*fp[2]+6*fp[3]

    @staticmethod
    def fp_bond3_value(fp):
        """Get the number of 3 bond sub-structures from fingerprint structure vector.

        Returns:
            int: the number of 3 bond sub-structures
        """
        return 1*fp[2]+4*fp[3]

    @staticmethod
    def fp_bond4_value(fp):
        """Get the number of 4 bond sub-structures from fingerprint structure vector.

        Returns:
            int: the number of 4 bond sub-structures
        """
        return 1*fp[3]


# -----------------------------------------------------------------------------
# ParticleSwarmOptimization: backend optimizer for feature estimation
# -----------------------------------------------------------------------------

class ParticleSwarmOptimization(object):
    """Numerical optimization as a backend optimizer of feature estimation.

    Attributes:
        evaluator (FeatureEvaluator): an evaluator of feature vector
        variable_data (array): feature vector of training data
        variable_range (array): min/max range of feasible feature vector
        variable_type (array): date type of variables
        struct_index(list): a list of feature vector indices for structure generation
        fixed_labels (list): a list of fixed component labels
        prediction_error (float, optional): acceptable range of prediction error
        extend_solution (bool): flag to extend the range of feature vector values
        use_data_ratio (float): ratio of training data used as initial value of a particle
        threshold (float): threshold of penalty values for acceptable solution
        num_particle (int): number of particles
        iteration (int): maximum number of iteration
        reset_interval (int): number of iteration before resetting all the particles
    """

    class Sampler(object):
        """Base class of sampling population for particle swarm optimization

        Attributes:
            vector_size (int): size of feature vector
            variable_range (array): min/max range of feasible feature vector
            variable_data (array): feature vector of training data
            use_data_ratio (float): ratio of training data used as initial value of a particle
            variable_err (array): acceptable numerical error of feature vector value
        """

        min_numerical_err = 0.00001
        min_numerical_err_ratio = 0.001

        def __init__(self, variable_range, variable_data, use_data_ratio):
            """Constructor of Sampler class

            Args:
            variable_range (array): min/max range of feasible feature vector
            variable_data (array): feature vector of training data
            use_data_ratio (float): ratio of training data used as initial value of a particle
            """
            self.vector_size = len(variable_range)
            self.variable_range = variable_range
            self.variable_data = variable_data
            self.use_data_ratio = use_data_ratio
            self.variable_err = self.min_numerical_err_ratio * (variable_range[:, 1] - variable_range[:, 0])
            for index in range(self.vector_size):
                self.variable_err[index] = max(self.variable_err[index], self.min_numerical_err)

        def get_population(self, num_sample):
            """Get particles of given number

            Args:
                num_sample (int): number of particles

            Returns:
                array:  vector of particles
            """
            samples = np.zeros((num_sample, len(self.variable_range)))
            for index in range(num_sample):
                samples[index] = self.sampling()
            return samples


    class RandomSampler(Sampler):
        """Sampling population from uniform distribution of variable ranges.

        Attributes:
            vector_size (int): size of feature vector
            variable_range (array): min/max range of feasible feature vector
            variable_data (array): feature vector of training data
            use_data_ratio (float): ratio of training data uused as initial value of a particle
        """

        def __init__(self, variable_range, variable_data, use_data_ratio):
            """Constructor of RandomSampler class

            Args:
                variable_range (array): min/max range of feasible feature vector
                variable_data (array): feature vector of training data
                use_data_ratio (float): ratio of training data used as initial value of a particle
            """
            super().__init__(variable_range, variable_data, use_data_ratio)

        def sampling(self):
            """Get one particle

            Returns:
                array: a particle
            """
            if self.use_data_ratio > random.uniform(0, 1):
                # use original data as initial position
                data = self.variable_data[random.randrange(0, len(self.variable_data))]
                for index in range(data.shape[0]):
                    if data[index] < self.variable_range[index][0]:
                        data[index] = self.variable_range[index][0]
                    elif data[index] > self.variable_range[index][1]:
                        data[index] = self.variable_range[index][1]
                return data
            else:
                # random data as initial position
                x_min = self.variable_range[:, 0]
                x_max = self.variable_range[:, 1]
                return x_min + np.random.rand(self.vector_size) * (x_max - x_min)

    class SolutionSampler(Sampler):
        """Sampling population from uniform distribution of solution space of
        linear regression function

        Attributes:
            variable_range (array): min/max range of feasible feature vector
            variable_data (array): feature vector of training data
            use_data_ratio (float): ratio of training data used as initial value of a particle
            evaluator (FeatureEvaluator): evaluator of a feature vector
            target_values (dict): a mapping of property and its target value
            sampling_method (str): a name of random walk method
            with_slack (bool): a flag of using slack variable to extend solution space with prediction error
            slack_size (int): the number of slack variables
            ns_var (array): matrix of the null space vectors
            ns_const (array): constants for the null space
            prj_matrix (array): matrix for projecting training data to solution plane
            feasible_matrix (array): matrix of the constraints on solution space
            points (array): current points of random walk in feature vector space
            ns_vectors (array): current points of random walk in solution space
            point_index (int): index of current point for next random walk
            base_point (array): an origin point on solution space for projection
        """

        default_start_points = 5
        """Default number of starting points of random walk"""

        default_internal_points = 5

        def __init__(self, variable_range, variable_data, use_data_ratio, evaluator, target_values, int_index,
                     method='hit_and_run'):
            """Constructor of SolutionSampler class

            Args:
                variable_range (array): min/max range of feasible feature vector
                variable_data (array): feature vector of training data
                use_data_ratio (float): ratio of training data used as initial value of a particle
                evaluator (FeatureEvaluator): evaluator of a feature vector
                target_values (dict): a mapping of property and its target value
                int_index (list): a list of flags of integer value of a feature vector
                method (str, optional): a name of random walk method. Defaults to 'hit_and_run'
            """
            super().__init__(variable_range, variable_data, use_data_ratio)
            self.infeasible = False
            self.evaluator = evaluator
            self.target_values = target_values
            self.int_index = int_index
            self.sampling_method = method
            self.with_slack = False
            self.slack_size = 0
            self.num_start_points = self.default_start_points
            try:
                self.ns_var, self.ns_const, self.prj_matrix = self.get_null_space(evaluator, target_values)
            except np.linalg.LinAlgError:
                logger.warning('equation matrix is singular, switch to ranodm sampling')
                self.infeasible = True
                return
            self.feasible_matrix = self.get_feasible_matrix(self.ns_var, self.ns_const)
            self.points, self.ns_vectors = \
                self.get_initial_points(self.num_start_points, self.feasible_matrix, self.ns_var)
            if self.infeasible:
                return
            self.point_index = -1
            self.base_point = self.points[0] @ self.ns_var + self.ns_const

        def get_population(self, num_sample):
            """Get particles of given number

            Args:
                num_sample (int): number of particles

            Returns:
                array:  vector of particles
            """
            samples = np.zeros((num_sample, len(self.variable_range) + self.slack_size))
            for index in range(num_sample):
                samples[index] = self.sampling()
            return samples

        def sampling(self):
            """Get one particle

            Returns:
                array: a particle
            """
            if self.use_data_ratio > random.uniform(0, 1):
                # use original data as initial position
                data = self.variable_data[random.randrange(0, len(self.variable_data))]
                for index in range(data.shape[0]):
                    if data[index] < self.variable_range[index][0]:
                        data[index] = self.variable_range[index][0]
                    elif data[index] > self.variable_range[index][1]:
                        data[index] = self.variable_range[index][1]
                if self.slack_size > 0:
                    data = np.append(data, np.zeros(self.slack_size))
                vector = self.base_point + self.prj_matrix @ (data - self.base_point)
                return vector
            else:
                if self.sampling_method == 'hit_and_run':
                    return self.hit_and_run()

        def hit_and_run(self):
            """Get a next point by hit_and_run random walk method

            Returns:
                array: a particle
            """
            vector_size = self.ns_var.shape[1]
            variable_size = vector_size - self.slack_size

            A_ub = self.feasible_matrix[:, :-1]
            b_ub = -1 * self.feasible_matrix[:, -1]
            vrange = self.variable_range
            self.point_index = (self.point_index + 1) % len(self.points)
            point = self.points[self.point_index]
            # find the both end points for random direction going through the point
            # A_ub @ (point + a * t) <= b_ub
            # -> (A_ub @ t) * a <= b_ub - A_ub @ point
            # direction = np.random.rand(ns_dim)
            # direction = np.random.normal(size=ns_dim)
            # direction = np.random.rand(vector_size)
            # uniform distribution on each dimension is not good
            # uniform sampling from d-sphere (Muller's method)
            try_count = 0
            while True:
                try_count += 1
                direction = np.random.normal(size=vector_size)
                direction = self.ns_var @ direction
                rhs = A_ub @ direction
                lhs = b_ub - A_ub @ point
                ub = np.min([l / r for (r, l) in zip(rhs, lhs) if r > 0])
                lb = np.max([l / r for (r, l) in zip(rhs, lhs) if r < 0])
                magnitude = lb + random.uniform(0, 1) * (ub - lb)
                new_point = point + magnitude * direction
                vector = (new_point @ self.ns_var + self.ns_const)
                # check the range of new point
                value_err = self.variable_err + 0.5 * np.array(self.int_index)
                check_vector = vector[:variable_size]
                if np.isnan(magnitude) or np.isnan(check_vector[0]) or \
                        np.any((vrange[:, 0] - value_err) > check_vector) or \
                        np.any(check_vector > (vrange[:, 1] + value_err)):
                    if try_count < 10:
                        # try hit-and-run again
                        continue
                    else:
                        # reset to an initial point
                        t = np.random.rand(len(self.ns_vectors))
                        new_point = (t / np.sum(t)) @ self.ns_vectors
                        vector = new_point @ self.ns_var + self.ns_const
                        self.points[self.point_index] = new_point
                        break
                else:
                    # new point after hit-and-run
                    self.points[self.point_index] = new_point
                    break
            return vector

        def get_null_space(self, evaluator, target_values):
            """Get null space vectors of solution space

            Args:
                evaluator (FeatureEvaluator): evaluator of a feature vector
                target_values (dict): a mapping of property and its target value

            Returns:
                array, array, array: null space vectors, null space constants, projection matrix
            """
            coefs = []
            consts = []
            coefs_slack = []
            consts_slack = []
            self.slack_size = 0
            coef_map = evaluator.get_coef()
            for prop, (coef, shift, std) in coef_map.items():
                target_value = target_values[prop]
                if target_value[0] == target_value[1]:
                    # no range in the target value
                    coefs.append(coef)
                    consts.append(target_value[0] + shift)
                else:
                    # set upper bound
                    if target_value[1] <= sys.float_info.max:
                        coefs_slack.append(coef)
                        consts_slack.append(target_value[1] + shift)
                        self.slack_size += 1
                    # set lower bound
                    if target_value[0] >= -sys.float_info.max:
                        coefs_slack.append(-coef)
                        consts_slack.append(-(target_value[0] + shift))
                        self.slack_size += 1
            if self.slack_size > 0:
                if len(coefs) > 0:
                    coefs = np.append(np.array(coefs), np.zeros(shape=(len(coefs), self.slack_size)), axis=1)
                    coefs_slack = np.append(np.array(coefs_slack), np.identity(self.slack_size), axis=1)
                    coefs = np.append(coefs, coefs_slack, axis=0)
                    consts = np.append(np.array(consts), np.array(consts_slack), axis=0)
                else:
                    coefs = np.append(np.array(coefs_slack), np.identity(self.slack_size), axis=1)
                    consts = np.array(consts_slack)
            else:
                coefs = np.array(coefs)
                consts = np.array(consts)
            u, s, vh = np.linalg.svd(coefs)
            # get one solution of coef * x = const
            sol = sp.linalg.solve(coefs @ vh[:len(consts)].T, consts)
            ns_const = sol @ vh[:len(consts)]
            # get null space
            ns_var = vh[len(consts):]
            # get projection matrix for training data
            vector_size = coefs.shape[1]
            prj_matrix = np.zeros(shape=(vector_size, vector_size))
            for vect in vh[:len(consts)]:
                vect0 = vect.reshape(1, len(vect))
                prj_matrix += vect0.T @ vect0
            prj_matrix = np.identity(len(prj_matrix)) - prj_matrix
            return ns_var, ns_const, prj_matrix

        def get_feasible_matrix(self, ns_var, ns_const):
            """Get matrix of constraints for solution space

            Args:
                ns_var (array): matrix of the null space vectors
                ns_const (array): constants for the null space

            Returns:
                array: matrix of the constraints on solution space
            """
            vector_size = ns_var.shape[1]
            variable_size = vector_size - self.slack_size
            ns_matrix = np.append(ns_var.T, ns_const.reshape(vector_size, 1), axis=1)

            # make a matrix for the constraints on variables
            # c_coef * X + c_const <= 0
            c_coefs = []
            c_consts = []
            # set constraint of variable range
            for index in range(variable_size):
                x_min = self.variable_range[index][0] - 0.5 * self.int_index[index]
                x_max = self.variable_range[index][1] + 0.5 * self.int_index[index]
                # x[index] >= x_min -> -1 * x[index] + xmin <= 0
                c_coef = np.zeros(vector_size)
                c_coef[index] = -1
                c_const = x_min
                c_coefs.append(c_coef)
                c_consts.append(c_const)
                # x[index] <= x_max -> 1 * x[index] - xmax <= 0
                c_coef = np.zeros(vector_size)
                c_coef[index] = 1
                c_const = -x_max
                c_coefs.append(c_coef)
                c_consts.append(c_const)
            # set constraint of slack variable
            for index in range(variable_size, variable_size + self.slack_size):
                x_min = 0
                # x[index] >= x_min -> -1 * x[index] + xmin <= 0
                c_coef = np.zeros(vector_size)
                c_coef[index] = -1
                c_const = x_min
                c_coefs.append(c_coef)
                c_consts.append(c_const)
            c_coefs = np.array(c_coefs)
            c_consts = np.array(c_consts)
            feasible_matrix = c_coefs @ ns_matrix
            feasible_matrix[:, -1] += c_consts
            return feasible_matrix

        def get_initial_points(self, num_points, feasible_matrix, ns_var):
            """Get staring points for random walk by affine combination of vertices of polytope

            Args:
                num_points (int): number of starting points
                feasible_matrix (array): matrix of the constraints on solution space
                ns_var (array): matrix of the null space vectors

            Returns:
                array, array: starting points in feature vector space and solution space
            """
            # get feasible points by solving linear program
            vector_size = ns_var.shape[1]
            A_ub = feasible_matrix[:, :-1]
            b_ub = -1 * feasible_matrix[:, -1]
            ns_vectors = []
            for index in range(min(vector_size, self.default_internal_points)):
                t = np.random.normal(size=vector_size)
                c = ns_var @ t
                opt_result = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                if opt_result.success:
                    ns_vectors.append(opt_result.x)
                else:
                    logger.warning('solution space of PSO is infeasible, switch to random sampling')
                    self.infeasible = True
                    break
                opt_result = sp.optimize.linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
                if opt_result.success:
                    ns_vectors.append(opt_result.x)
                else:
                    logger.warning('solution space of PSO is infeasible, switch to random sampling')
                    self.infeasible = True
                    break
            if self.infeasible:
                return [], []
            ns_vectors = np.array(ns_vectors)
            # make feasible points
            points = []
            for index in range(num_points):
                t = np.random.rand(len(ns_vectors))
                point = (t / np.sum(t)) @ ns_vectors
                points.append(point)
            return points, ns_vectors

    random_walk_sampling = True
    """Flag to use random walk for sampling particles"""

    invalid_score = 1000
    """Maximum penalty score for bad cases"""

    @classmethod
    def use_random_walk(cls, value):
        """Set a flag of using random walk for sampling particles

        Args:
            value (bool): true if random walk is used
        """
        cls.random_walk_sampling = value

    def __init__(self, evaluator, variable_data, variable_range, variable_type, struct_index, fix_condition=None,
                 num_candidate=1, prediction_error=1.0, extend_solution=False,
                 use_data_ratio=0.1, threshold=1.0, num_particle=1000, iteration=1000, reset_interval=10):
        """Constructor of ParticleSwarmOptimization.

        Args:
            evaluator (FeatureEvaluator): an evaluator of feature vector
            variable_data (array): feature vector of actual data
            variable_range (array): min/max range of feasible feature vector
            variable_type (array): data type of variables
            struct_index (list): a list of feature vector indices for structure generation
            fix_condition(ComponentFixCondition, optional): fix condition. Defaults to None.
            prediction_error (float, optional): acceptable range of prediction error
            extend_solution (bool, optional): flag to extend the range of feature vector values Defaults to False.
            use_data_ratio (float, optional): ratio of actual data used as initial value of a particle. Defaults to 0.1.
            threshold (float, optional): threshold of penalty values for acceptable solution. Defaults to 1.0.
            num_particle (int, optional): number of particles. Defaults to 1000.
            iteration (int, optional): maximum number of iteration. Defaults to 1000.
            reset_interval (int, optional): number of iteration before resetting all the particles. Defaults to 10.
        """
        self.evaluator = evaluator
        self.variable_data = variable_data
        self.variable_range = variable_range
        self.variable_type = variable_type
        self.struct_index = struct_index
        self.fixed_labels = []
        self.num_candidate = num_candidate
        self.prediction_error = prediction_error
        self.extend_solution = extend_solution
        self.use_data_ratio = use_data_ratio
        self.threshold = threshold
        self.num_particle = num_particle
        self.iteration = iteration * len(self.evaluator.get_target_properties())
        self.reset_interval = reset_interval  # * len(self.evaluator.get_target_properties())
        if fix_condition is not None:
            self.fixed_labels = fix_condition.get_labels()

    def rounding(self, data, int_index, sampler):
        """Round data to integer

        Args:
            data (np.array): feature vector value
            int_index (list): flag of integer index
            sampler (ParticleSwarmOptimization.Sampler): particle sampler

        Returns:
             np.array: rounded feature vector value
        """
        # get slack size
        slack_size = 0
        if isinstance(sampler, ParticleSwarmOptimization.SolutionSampler):
            slack_size = sampler.slack_size

        if slack_size > 0:
            data_round = np.array(data[:, 0: -slack_size])
            data_round[:, int_index] = np.round(data[:, 0: -slack_size][:, int_index])
        else:
            data_round = np.array(data)
            data_round[:, int_index] = np.round(data[:, int_index])
        return data_round

    def in_taboo_list(self, rposition, taboo_list):
        """Check if a particle in a taboo list

        Args:
            rposition (np.array): a rounded particle
            taboo_list (list): a list of taboo particle

        Returns:
            bool: true if a particle in a taboo list
        """
        st_position = rposition[self.struct_index]
        for taboo in taboo_list:
            st_taboo = taboo[self.struct_index]
            if np.all(st_taboo <= st_position) and np.all(st_position <= st_taboo):
                return True
        return False

    def particles_initialize(self, sampler, int_index):
        """Initialize particle position.

        Args:
            sampler (Sampler): particle sampler
            int_index (list): list of integer flags

        Returns:
            array, array: particle position and particle velocity
        """
        # generate population avoiding particles in taboo_list
        position = sampler.get_population(self.num_particle)
        rposition = self.rounding(position, int_index, sampler)
        velocity = np.zeros(shape=position.shape)
        for index in range(len(position)):
            ref_index = random.randrange(0, len(position))
            velocity[index] = random.uniform(0, 1) * (position[ref_index] - position[index])
        return position, rposition, velocity

    def evaluate_position(self, target_values, rposition, taboo_list):
        """Evaluate penalties of particle positions.

        Args:
            target_values (dict): a mapping of target property and its values
            rposition (array): rounded positions of particles
            taboo_list (list): already found solution

        Returns:
            array, array, array, array : total error, prediction, prediction error and constraint violation
        """
        vrange = self.variable_range
        scores = self.evaluator.evaluate(target_values, rposition, self.prediction_error, self.fixed_labels)
        for index in range(len(scores)):
            rvector = rposition[index]
            if any([not vrange[jdx][0] <= rvector[jdx] <= vrange[jdx][1] for jdx in range(len(vrange))]):
                scores[index] = self.invalid_score
            elif self.in_taboo_list(rvector, taboo_list):
                scores[index] = self.invalid_score
        return scores

    def update_position(self, position, velocity, int_index, sampler):
        """Update particle position with velocity.

        Args:
            position (array): positions of particles
            velocity (array): velocities of particles
            int_index (list): list of integer flags
            sampler (ParticleSwarmOptimization.Sampler): particle sampler

        Returns:
            array: new positions of particles
        """
        position += velocity
        rposition = self.rounding(position, int_index, sampler)
        return position, rposition

    def update_velocity(self, position, velocity, personal_best, global_best,
                        w=0.8, ro_max1=0.9, ro_max2=0.6):
        """Update velocities of particles.

        Args:
            position (array): positions of particles
            velocity (array): velocities of particles
            personal_best (array): best position of each particle
            global_best (float): best position among all the particles
            w (float, optional): weight of current velocity. Defaults to 0.9.
            ro_max1 (float, optional): weight for the direction to personal best. Defaults to 0.9.
            ro_max2 (float, optional): weight for the direction to global best. Defaults to 0.6.

        Returns:
            array: new velocities of particles
        """
        ro1 = np.random.rand(len(position)) * ro_max1
        ro2 = np.random.rand(len(position)) * ro_max2
        velocity *= w
        velocity += ((1.0-w) * ro1) @ (personal_best - position)
        for index in range(len(position)):
            velocity[index] += ((1.0-w) * ro2[index]) * (global_best - position[index])
        return velocity

    def optimize(self, target_values, num_candidate, old_estimates, verbose=True):
        """Find a feature vector minimizing the estimation error and penalty for feature value feasibility.

        Args:
            target_values (dict): a mapping of target property and its value
            num_candidate (int): number of candidate feature vector to find.
            old_estimates (list): an existing estimates
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            list, list, float: a list of feature vector, a list of feature vector ranges, score
        """
        if verbose:
            print('search feature vectors by particle swarm optimization')

        # initialize particle and velocity
        candidates = [o for o in old_estimates]
        int_index = [x == int for x in self.variable_type]
        failed = []
        # get sampler for particles
        if self.random_walk_sampling and self.evaluator.is_linear_model():
            sampling_method = 'hit_and_run'
            sampler = self.SolutionSampler(self.variable_range, self.variable_data, self.use_data_ratio,
                                           self.evaluator, target_values, int_index, method=sampling_method)
            if sampler.infeasible:
                logger.info(' : vector size={0}'.format(len(self.variable_range)))
                sampler = self.RandomSampler(self.variable_range, self.variable_data, self.use_data_ratio)
            else:
                logger.info(' ({0} walk): vector size={1}'.format(sampling_method, len(self.variable_range)))
        else:
            logger.info(' : vector size={0}'.format(len(self.variable_range)))
            sampler = self.RandomSampler(self.variable_range, self.variable_data, self.use_data_ratio)

        position, rposition, velocity = self.particles_initialize(sampler, int_index)

        # keep best score
        personal_best_positions = position
        personal_best_rpositions = rposition
        personal_best_scores = self.evaluate_position(target_values, rposition, candidates + failed)
        best_particle = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_particle]
        global_best_rposition = personal_best_rpositions[best_particle]
        global_best_score = personal_best_scores[best_particle]

        raw_candidate = global_best_position
        candidate = global_best_rposition
        candidate_score = global_best_score

        solution_count = 0
        last_update = -1
        candidates_range = []
        scores = []
        iteration_count = 0
        total_iteration_count = 0
        while iteration_count < self.iteration:
            iteration_count += 1
            total_iteration_count += 1

            if logger.isEnabledFor(logging.INFO):
                logger.info('pso iteration=%d score=%f', total_iteration_count, global_best_score)
            if logger.isEnabledFor(logging.DEBUG):
                for index in range(len(position)):
                    logger.debug(self.evaluator.detail_info(target_values, rposition[index], position[index],
                                                            self.fixed_labels))

            # update particle velocity
            self.update_velocity(position, velocity, personal_best_positions, global_best_position)
            # update particle position
            position, rposition = self.update_position(position, velocity, int_index, sampler)

            # update personal best
            new_score = self.evaluate_position(target_values, rposition, candidates + failed)
            for idx in range(self.num_particle):
                if new_score[idx] < personal_best_scores[idx]:
                    personal_best_scores[idx] = new_score[idx]
                    personal_best_positions[idx] = position[idx]
                    personal_best_rpositions[idx] = rposition[idx]

            # update global best
            best_particle = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[best_particle]
            global_best_rposition = personal_best_rpositions[best_particle]
            global_best_score = personal_best_scores[best_particle]
            if logger.isEnabledFor(logging.INFO):
                logger.info('best_score={0} candidate={1}'.format(global_best_score, candidate_score))

            # update best solution
            if global_best_score < candidate_score:
                raw_candidate = global_best_position
                candidate = global_best_rposition
                candidate_score = global_best_score
                last_update = total_iteration_count
                continue

            # check if enough update
            if total_iteration_count - last_update >= self.reset_interval:
                if candidate_score < self.threshold:
                    # no better solution, keep current candidate
                    if self.extend_solution:
                        candidate_range = self.extend_variable_range(target_values, candidate)
                    else:
                        candidate_range = np.array([candidate, candidate])
                    candidates.append(candidate)
                    candidates_range.append(candidate_range)
                    scores.append(candidate_score)
                    solution_count += 1
                    iteration_count = 0
                    logger.info('new solution ({0}/{1}): iteration={2:d} score={3:.3f}'.
                          format(solution_count, num_candidate, total_iteration_count+1, candidate_score))
                    if verbose:
                        logger.info(self.evaluator.detail_info(target_values, candidate, raw_candidate, self.fixed_labels))
                    # terminate if enough solutions
                    if solution_count >= num_candidate:
                        break
                else:
                    if verbose:
                        logger.info('reset particle (current best): iteration={0:d} score={1:.3f}'.
                              format(total_iteration_count+1, candidate_score))
                        logger.info(self.evaluator.detail_info(target_values, candidate, raw_candidate, self.fixed_labels))

                # reset particle positions
                position, rposition, velocity = self.particles_initialize(sampler, int_index)
                # keep best score
                personal_best_positions = position
                personal_best_rpositions = rposition
                personal_best_scores = self.evaluate_position(target_values, rposition, candidates + failed)
                best_particle = np.argmin(personal_best_scores)
                global_best_position = personal_best_positions[best_particle]
                global_best_rposition = personal_best_rpositions[best_particle]
                global_best_score = personal_best_scores[best_particle]
                raw_candidate = global_best_position
                candidate = global_best_rposition
                candidate_score = global_best_score
                last_update = total_iteration_count

        if verbose:
            if solution_count == 0:
                print('no candidates are found')
            else:
                print('{0} candidates are found'.format(solution_count))

        return candidates[len(old_estimates):], [c.T for c in candidates_range], scores

    def extend_variable_range(self, target_values, solution):
        """Extend the range of feature value as far as prediction error is within standard deviation.

        Args:
            target_values (dict): a mapping of target property and its value
            solution (array): feature vector

        Returns:
            array: feature vector min/max range
        """
        solution_range = np.asarray([solution, solution])
        solution_data = np.asarray([solution])
        for index in range(len(solution)):
            if self.variable_type[index] == int:
                original = solution_data[0][index]
                # find lower bound
                for new_val in range(int(original)-1, int(self.variable_range[index][0]), -1):
                    solution_data[0][index] = new_val
                    score = self.evaluator.evaluate(target_values, solution_data,
                                                    self.prediction_error, self.fixed_labels)
                    if score[0] < self.threshold:
                        solution_range[0][index] = new_val
                    else:
                        break
                # find upper bound
                for new_val in range(int(original)+1, int(self.variable_range[index][1]), 1):
                    solution_data[0][index] = new_val
                    score = self.evaluator.evaluate(target_values, solution_data,
                                                    self.prediction_error, self.fixed_labels)
                    if score[0] < self.threshold:
                        solution_range[1][index] = new_val
                    else:
                        break
                solution_data[0][index] = original
        return solution_range
