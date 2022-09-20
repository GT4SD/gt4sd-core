# -*- coding:utf-8 -*-
"""
Generation.py

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
from .FeatureEstimation import *
from .ChemGenerator.ChemGraph import AtomGraph, ChemVertex
from .ChemGenerator.ChemGraphGen import *

import numpy as np

from collections import Counter

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# MoleculeGenerator: molecule generator from an estimated feature vector
# -----------------------------------------------------------------------------

class MoleculeGenerator(object):
    """Molecule generator for given feature vector estimated by FeatureEstimator.

    Attributes:
        label (str): label component
        evaluator (FeatureEvaluator): a evaluator of merged feature vector
    """

    ring_graph_list = None
    """list: default candidates of ring structures"""

    def __init__(self, label, evaluator):
        """Constructor of MoleculeGenerator.

        Args:
            label (str): label component
            evaluator (FeatureEvaluator): a evaluator of merged feature vector
        """
        self.label = label
        self.evaluator = evaluator
        # set default ring list
        if self.ring_graph_list is None:
            self.ring_graph_list = self.read_rings()

    @staticmethod
    def read_rings(filename=None):
        """Read ring definitions

        Args:
            filename (str, optional): filename of ring file. Default to None.

        Returns:
            list: list of ring smiles
        """
        if filename is None:
            # use default structural rule (config/default_rings.csv)
            src_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(src_dir, 'config/default_rings.csv')

        if not os.path.exists(filename):
            logger.error('no structural rule file:{0}'.format(filename))
            return []

        dataframe = pd.read_csv(filename)
        ring_list = []
        for index in range(len(dataframe)):
            ring_list.append(dataframe['SMILES'][index])
        return ring_list

    def generate_molecule(self, design_param, fix_condition, feature_estimate, candidates,
                          max_solution=0, max_node=0, max_depth=0, beam_size=0, verbose=True):
        """Generate molecules satisfying give feature vector.

        Args:
            design_param (DesignParam): design parameter
            fix_condition (ComponentFixCondition): fixed components.
            feature_estimate (FeatureEstimateResult): feature estimation results
            candidates (list): a list of FeatureEstimationResult.Candidate objects
            max_solution (int, optional): maximum number of solutions to find. Defaults to 0.
            max_node (int, optional): maximum number of search tree nodes to search. Defaults to 0.
            max_depth (int, optional): maximum depth of iterative deepening. Defaults to 0.
            beam_size (int, optional): beam size of beam search. Defaults to 0.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            list: a list of FeatureEstimationResults.Candidate
        """
        moldata = self.evaluator.get_moldata().get_subdata(self.label)
        features_list = self.evaluator.get_label_features_list(self.label)
        selection_mask = self.evaluator.get_label_selection_mask(self.label)
        max_atom = self.evaluator.get_label_max_atom(self.label)
        max_ring = self.evaluator.get_label_max_ring(self.label)
        atom_range_map = self.evaluator.get_label_atom_range(self.label)
        ring_range_map = self.evaluator.get_label_ring_range(self.label)
        aring_range_map = self.evaluator.get_label_aring_range(self.label)
        atom_valence_map = self.evaluator.get_label_atom_valence(self.label)
        atom_range = dict()
        for atom, arange in atom_range_map.items():
            atom_range[atom] = [arange[0], arange[1]]
        ring_range = dict()
        for ring, rrange in ring_range_map.items():
            ring_range[ring] = [rrange[0], rrange[1]]
        aring_range = dict()
        for aring, arrange in aring_range_map.items():
            aring_range[aring] = [arrange[0], arrange[1]]
        atom_valence = dict()
        for atom, v in atom_valence_map.items():
            (base_atom, charge, valence) = ChemVertex.register_vertex(atom)
            if v > valence:
                atom_valence[atom] = v

        # generate molecules
        gen_candidates = dict()
        for candidate in candidates:
            if candidate.get_molecules() is None:
                # check if candidate is duplicated
                if candidate.get_duplicate():
                    candidate.set_molecules([])
                    continue
                # check if zero vector
                if sum(candidate.get_selection_mask()) == 0:
                    candidate.set_molecules([])
                    continue
                gen_candidates[candidate.get_id()] = candidate

        if len(gen_candidates) == 0:
            logger.info('no new feature vector candidates for {0} (duplicated or size zero?)'.format(self.label))
            return []

        # create molecule evaluator
        molecule_evaluator = None
        if self.evaluator.has_single_target_label() and \
                self.evaluator.has_only_online_feature():
            # prepare molecule evaluator and null candidate for evaluating generated molecule
            # by a regression model
            feature_dtype = self.evaluator.get_feature_dtype(with_mask=False)
            feature_range = self.evaluator.get_feature_range(with_mask=False)
            selection_mask = self.evaluator.get_selection_mask()
            molecule_evaluator = MoleculeEvaluator(self.evaluator, design_param, self.label)
            null_candidate_id = FeatureEstimator.get_time_stamp_id()
            null_candidate = FeatureEstimationResult.Candidate(null_candidate_id, self.label, feature_estimate,
                                                               None, None,
                                                               feature_dtype, feature_range, selection_mask, 0)

        # generate molecules from a candidate feature estimate
        range_params = design_param.get_range_params()
        prediction_error = range_params['prediction_error']
        resource_constraints = []
        for index, candidate in enumerate(gen_candidates.values()):
            # get feature vector and feature value range
            feature_vector = candidate.get_feature_vector()
            feature_range = candidate.get_feature_range(with_mask=False)
            feature_vector_estimate = candidate.get_whole_feature_vector()

            # generate molecules by SMILES
            logger.info('feature vector {0} ({1}/{2}) size={3}: max_atom={4} max_ring={5}'
                  .format(candidate.get_id(), index + 1, len(gen_candidates), sum(selection_mask),
                          max_atom, max_ring))
            
            resource_constraint = self.get_resource_constraint_from_feature_list(candidate.get_id(), moldata,
                                                                                 feature_vector, feature_range,
                                                                                 features_list, selection_mask,
                                                                                 max_atom, max_ring,
                                                                                 atom_range, ring_range,
                                                                                 aring_range, atom_valence,
                                                                                 prediction_error, molecule_evaluator)
            resource_constraints.append(resource_constraint)

        # multiple resource constraint
        if len(resource_constraints) == 1 and \
                isinstance(resource_constraints[0], ChemGraphConstraintByRegressionFunction):
            generator = ChemGraphGenerator(resource_constraints[0], verbose=verbose)
        else:
            resource_constraint_set = ChemGraphConstraintSet(resource_constraints,
                                                             prediction_error, molecule_evaluator)
            generator = ChemGraphGenerator(resource_constraint_set, verbose=verbose)

        # generate chemical graphs
        generator.search(max_solution=max_solution, max_node=max_node, max_depth=max_depth,
                         beam_size=beam_size)

        # extract generation results from generator
        solution = generator.get_solution()
        result_candidates = []
        for candidate_id, candidate in gen_candidates.items():
            sols = solution[candidate_id]
            # generate molecule object from smiles
            molecules = []
            for index, (g, path) in enumerate(sols):
                molecule = GeneratedMolecule(index, g, GenerationPath.op_sequence_to_str(path),
                                             candidate, moldata.get_mol_type())
                molecules.append(molecule)

            # store generated molecules to candidate object
            candidate.set_molecules(molecules)
            candidate.set_feature_extracted(False)

            if len(molecules) > 0:
                result_candidates.append(candidate)

        # extract generation results for null candidate
        if '' in solution:
            sols = solution['']
            # generate molecule object from smiles
            molecules = []
            for index, (g, path) in enumerate(sols):
                molecule = GeneratedMolecule(index, g, GenerationPath.op_sequence_to_str(path),
                                             null_candidate, moldata.get_mol_type())
                molecules.append(molecule)

            # store generated molecules to candidate object
            null_candidate.set_molecules(molecules)
            null_candidate.set_feature_extracted(False)

            if len(molecules) > 0:
                result_candidates.append(null_candidate)
                feature_estimate.add_candidate(design_param, fix_condition, null_candidate, [])

        return result_candidates

    def get_resource_constraint_from_feature_list(self, constraint_id, moldata,
                                                  feature_vector, feature_range, features_list, selection_mask,
                                                  max_atom, max_ring, 
                                                  atom_range_map, ring_range_map, aring_range_map, atom_valence_map,
                                                  prediction_error=1.0, molecule_evaluator=None):
        """Get resource constraint from specified features in a feature set.

        Args:
            constraint_id (str): id of resource constraint
            moldata (MolData): moldata object
            feature_vector (array): feature vectors by min/max ranges of feature vector values
            feature_range (array): ranges of feature vector value
            features_list (list): list of feature sets corresponding to the feature vector
            selection_mask (list): list of flags of feature value selection
            max_atom (int): maximum number of heavy atom.
            max_ring (int): maximum number of rings.
            atom_range_map (dict): min/max range of atom symbols
            ring_range_map (dict): min/max range of ring sizes
            aring_range_map (dict): min/max range or aromatic ring sizes
            atom_valence_map (dict): atom valence
            prediction_error (float): acceptable range of prediction error
            molecule_evaluator (MoleculeEvaluator): molecule feature vector evaluator. Defaults None.

        Returns:
            ChemGraphConstraint: a resource constraint object
        """
        # change feature vector to an array of feature value range
        if feature_vector is not None and len(feature_vector.shape) == 1:
            feature_vector = np.asarray([feature_vector, feature_vector])
            feature_vector = feature_vector.T

        # get feature values for construction
        complete_atoms = list(atom_range_map.keys())
        complete_rings = list(ring_range_map.keys())
        complete_aromatics = list(aring_range_map.keys())
        atoms = dict()
        rings = dict()
        aromatics = dict()
        fragments = dict()

        # get feature value from selected feature vector
        index = 0
        data_index = 0
        online_features = defaultdict(list)
        for features in features_list:
            for feature in features.get_feature_list():
                if isinstance(feature, HeavyAtomExtractor.Feature):
                    for idx in range(feature.get_vector_size()):
                        if selection_mask[index+idx]:
                            if feature_vector is None:
                                atoms[feature.get_index()] = feature_range[index+idx].astype(int)
                            else:
                                atoms[feature.get_index()] = feature_vector[data_index].astype(int)
                            data_index += 1
                elif isinstance(feature, RingExtractor.Feature):
                    for idx in range(feature.get_vector_size()):
                        if selection_mask[index+idx]:
                            if feature_vector is None:
                                rings[feature.get_index()] = feature_range[index+idx].astype(int)
                            else:
                                rings[feature.get_index()] = feature_vector[data_index].astype(int)
                            data_index += 1
                elif isinstance(feature, AromaticRingExtractor.Feature):
                    for idx in range(feature.get_vector_size()):
                        if selection_mask[index+idx]:
                            if feature_vector is None:
                                aromatics[feature.get_index()] = feature_range[index+idx].astype(int)
                            else:
                                aromatics[feature.get_index()] = feature_vector[data_index].astype(int)
                            data_index += 1
                elif isinstance(feature, FingerPrintStructureExtractor.Feature):
                    for idx in range(feature.get_vector_size()):
                        if selection_mask[index + idx]:
                            if feature_vector is None:
                                fragments[feature] = feature_range[index + idx].astype(int)
                            else:
                                fragments[feature] = feature_vector[data_index].astype(int)
                            data_index += 1
                else:
                    # other features has no built-in counter in the structure generation
                    num_selection = sum(selection_mask[index:index+feature.get_vector_size()])
                    if features.is_online_update() and num_selection > 0:
                        online_features[features].append(feature)
                    data_index += num_selection
                index += feature.get_vector_size()

        # get default ranges of features from training data
        data_atom_range = atom_range_map
        data_ring_range = ring_range_map
        data_aromatic_range = aring_range_map

        # print give feature vector
        atom_str = ' * heavy atom:{'
        for atom, val in atoms.items():
            if val[0] == val[1]:
                atom_str += '{0}:[{1:d}]'.format(atom, val[0])
            else:
                atom_str += '{0}:[{1:d},{2:d}]'.format(atom, val[0], val[1])
            if atom in atom_valence_map:
                atom_str += ':{0} '.format(atom_valence_map[atom])
            else:
                atom_str += ' '
        if len(data_atom_range) > len(atoms):
            atom_str = atom_str.strip()
            atom_str += '} {'
            for atom, val in data_atom_range.items():
                if atom not in atoms:
                    atom_str += '{0}:[{1:d},{2:d}]'.format(atom, val[0], val[1])
                    if atom in atom_valence_map:
                        atom_str += ':{0} '.format(atom_valence_map[atom])
                    else:
                        atom_str += ' '
        atom_str = atom_str.strip()
        atom_str += '}'
        ring_str = ' * ring:{'
        for ring, val in rings.items():
            if val[0] == val[1]:
                ring_str += '{0}:[{1:d}] '.format(ring, val[0])
            else:
                ring_str += '{0}:[{1:d},{2:d}] '.format(ring, val[0], val[1])
        if len(data_ring_range) > len(rings):
            ring_str = ring_str.strip()
            ring_str += '} {'
            for ring, val in data_ring_range.items():
                if ring not in rings:
                    ring_str += '{0}:[{1:d},{2:d}] '.format(ring, val[0], val[1])
        ring_str = ring_str.strip()
        ring_str += '}'
        aromatic_str = ' * aromatic:{'
        for aromatic, val in aromatics.items():
            if val[0] == val[1]:
                aromatic_str += '{0}:[{1:d}] '.format(aromatic, val[0])
            else:
                aromatic_str += '{0}:[{1:d},{2:d}] '.format(aromatic, val[0], val[1])
        if len(data_aromatic_range) > len(aromatics):
            aromatic_str = aromatic_str.strip()
            aromatic_str += '} {'
            for aromatic, val in data_aromatic_range.items():
                if aromatic not in aromatics:
                    aromatic_str += '{0}:[{1:d},{2:d}] '.format(aromatic, val[0], val[1])
        aromatic_str = aromatic_str.strip()
        aromatic_str += '}'
        fragment_str = ' * fragment:{'
        for fragment, val in fragments.items():
            if val[0] == val[1]:
                fragment_str += '{0}:[{1:d}] '.format(fragment.get_id(), val[0])
            else:
                fragment_str += '{0}:[{1:d},{2:d}] '.format(fragment.get_id(), val[0], val[1])
        fragment_str = fragment_str.strip()
        fragment_str += '}'
        if len(online_features) > 0:
            online_str = ' * features:{'
            for features, feature_list in online_features.items():
                online_str += '{0}:['.format(features.get_id())
                for feature in feature_list:
                    online_str += '{0} '.format(feature.get_id())
                online_str = online_str.strip()
                online_str += '] '
            online_str = online_str.strip()
            online_str += '}'

        # get additional atom from sub-structures
        missing_atom = Counter()
        root_atom = Counter()
        root_aromatic_atom = Counter()
        for feature, count in fragments.items():
            if count[0] > 0:
                atom_count = feature.get_graph().get_atom_count()
                if isinstance(feature, FingerPrintStructureExtractor.Feature):
                    if feature.is_aromatic_root_atom():
                        root_aromatic_atom[feature.get_root_atom()] += count[0]
                    root_atom[feature.get_root_atom()] += count[0]
                # count atoms not in atoms
                for atom in atom_count:
                    if atom not in atoms:
                        missing_atom[atom] = max(missing_atom[atom], atom_count[atom])
        # update missing_atom by root_atom counter
        for atom in missing_atom:
            missing_atom[atom] = max(missing_atom[atom], root_atom[atom])

        # get addition atom not specified explicitly
        additional_atoms = dict()
        if len(atoms) > 0:
            mandatory_atoms = sum(atoms.values())[0]
        else:
            mandatory_atoms = 0
        for atom in complete_atoms:
            if atom not in atoms:
                if atom in missing_atom:
                    num_atom = missing_atom[atom]
                    additional_atoms[atom] = np.array([min(num_atom, max_atom - mandatory_atoms),
                                                       max_atom - mandatory_atoms], dtype=int)
                else:
                    additional_atoms[atom] = np.array([0, max_atom - mandatory_atoms], dtype=int)

        # set atom resource
        num_atom = np.zeros(2, dtype=int)
        num_atom += sum(atoms.values())
        num_atom += sum(additional_atoms.values())
        if max_atom < num_atom[0]:
            total_atom_range = [num_atom[0], num_atom[1]]
            logger.warning('wrong atom range:[%d, %d] max_atom=%d', num_atom[0], num_atom[1], max_atom)
        else:
            total_atom_range = [num_atom[0], min(num_atom[1], max_atom)]

        atom_resource = {}
        for atom, count in atoms.items():
            arange = list(count)
            atom_resource[atom] = arange
        for atom, count in additional_atoms.items():
            if atom in data_atom_range:
                drange = data_atom_range[atom]
                if drange[1] < count[0]:
                    atom_resource[atom] = list(count)
                    logger.warning('wrong atom %s range:[%d, %d] data range[%d, %d]',
                                   atom, count[0], count[1], drange[0], drange[1])
                else:
                    arange = [max(count[0], drange[0]), min(count[1], drange[1])]
                    # add valence to atom resource
                    atom_resource[atom] = arange
            else:
                atom_resource[atom] = list(count)

        # get ring patters for generation
        ring_counter = Counter()
        for ring in complete_rings:
            if ring in rings:
                ring_counter[ring] = rings[ring][1]
            else:
                ring_counter[ring] = max_ring
        aromatic_counter = Counter()
        for ring in complete_aromatics:
            if ring in aromatics:
                aromatic_counter[ring] = aromatics[ring][1]
            elif ring in rings:
                aromatic_counter[ring] = rings[ring][1]
            else:
                aromatic_counter[ring] = max_ring

        # set ring range
        ring_range = dict()
        ring_minmax = np.zeros(2, dtype=int)
        for ring, count in rings.items():
            ring_range[ring] = list(count)
            ring_minmax += count
        for ring in complete_rings:
            if ring not in rings:
                if ring in data_ring_range:
                    ring_range[ring] = list(data_ring_range[ring])
                    ring_minmax += data_ring_range[ring]
                else:
                    ring_range[ring] = [0, ring_counter[ring]]
        total_ring_range = [min(ring_minmax[0], max_ring), max_ring]
        aromatic_range = {}
        aromatic_minmax = np.zeros(2, dtype=int)
        for ring, count in aromatics.items():
            aromatic_range[ring] = list(count)
            aromatic_minmax += count
        for ring in complete_aromatics:
            if ring not in aromatics:
                if ring in data_aromatic_range:
                    aromatic_range[ring] = list(data_aromatic_range[ring])
                    aromatic_minmax += data_aromatic_range[ring]
                else:
                    aromatic_range[ring] = [0, aromatic_counter[ring]]
                if ring in rings:
                    aromatic_range[ring][1] = min(aromatic_range[ring][1], rings[ring][1])
        total_aromatic_range = [min(aromatic_minmax[0], max_ring), max_ring]

        # get ring patters for generation
        ring_patterns = \
            MoleculeGenerator.get_ring_patterns(total_ring_range, ring_range,
                                                total_aromatic_range, aromatic_range,
                                                total_atom_range, atom_resource, self.ring_graph_list)

        # set ring resource
        ring_resource = {}
        replacement = {}
        # set atom replacement
        # get atoms used in rings
        replace_atoms = moldata.get_ring_replacement()
        for atom, count in replace_atoms.items():
            if atom in data_atom_range:
                replacement[atom] = [0, min(count, data_atom_range[atom][1])]
            else:
                replacement[atom] = [0, count]

        for smiles, max_use in ring_patterns.items():
            if len(replacement) > 0:
                t_replacement = tuple({s: tuple(r) for s, r in replacement.items()}.items())
                ring_resource[(smiles, t_replacement)] = [0, max_use]
            else:
                ring_resource[smiles] = [0, max_use]

        # print basic resource from feature vector
        logger.info(atom_str)
        ring_str += ' replace:{'
        for atom, val in replacement.items():
            if val[0] == val[1]:
                ring_str += '{0}:[{1:d}] '.format(atom, val[0])
            else:
                ring_str += '{0}:[{1:d},{2:d}] '.format(atom, val[0], val[1])
        ring_str = ring_str.strip()
        ring_str += '}'
        logger.info(ring_str)
        logger.info(aromatic_str)
        logger.info(fragment_str)
        if len(online_features) > 0:
            logger.info(online_str)

        # set fragment constraints
        fragment_const = {}
        for fragment, count in fragments.items():
            fragment_const[fragment.get_fragment()] = list(count)

        # add valence to atom resource
        new_atom_resource = {}
        for atom, arange in atom_resource.items():
            if atom in atom_valence_map:
                new_atom_resource[(atom, atom_valence_map[atom])] = arange
            else:
                new_atom_resource[atom] = arange
        atom_resource = new_atom_resource

        # create generator object
        if feature_vector is None:
            resource_constraint = ChemGraphConstraintByRegressionFunction(constraint_id, atom_resource,
                                                                          ring_resource, fragment_const,
                                                                          online_features, ring_range,
                                                                          aromatic_range, total_atom_range,
                                                                          total_ring_range, total_aromatic_range,
                                                                          prediction_error, molecule_evaluator)
        else:
            resource_constraint = ChemGraphConstraint(constraint_id, atom_resource, ring_resource,
                                                      fragment_const, online_features, ring_range,
                                                      aromatic_range, total_atom_range, total_ring_range,
                                                      total_aromatic_range, prediction_error,
                                                      molecule_evaluator)

        return resource_constraint

    @staticmethod
    def get_ring_patterns(total_ring_range, ring_range, total_aromatic_range, aromatic_range,
                          total_atom_range, atom_resource, rings):
        """Get appropriate rings registered in AtomGraph.

        Args:
            total_ring_range (array): range of the number of total rings
            ring_range (dict): range of the number of rings of each size
            total_aromatic_range (array): range of the number of total aromatic rings
            aromatic_range (dict): range of the number of aromatic rings of each size
            total_atom_range (array): range of the number of total atoms
            atom_resource (dict): range of number of available atoms
            rings (list, optional): a list of smiles of rings. Defaults to None

        Returns:
            dict: appropriate rings with their maximum usage
        """
        good_rings = {}
        for smiles in rings:
            # check if registered rings is acceptable
            if isinstance(smiles, str):
                graph = AtomGraph(smiles=smiles)
            elif isinstance(smiles, AtomGraph):
                graph = smiles
            else:
                continue
            atom_count = graph.get_atom_count()
            max_ring_use = int(total_atom_range[1]/graph.num_vertices())
            # check atom resource
            enough_atom_resource = True
            for atom, count in atom_count.items():
                if atom is ChemVertex.wild_card_atom:
                    continue
                if atom not in atom_resource:
                    # no atom resource
                    enough_atom_resource = False
                    break
                else:
                    if atom != 'C':
                        max_ring_use = min(max_ring_use, int(atom_resource[atom][1] / count))
                        if count > atom_resource[atom][1]:
                            # not enough atom
                            enough_atom_resource = False
                            break
            if not enough_atom_resource:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('ring_pattern: skip %s due to atom resource', graph.to_smiles())
                continue
            graph_ring_counter, graph_aromatic_counter = graph.count_rings_sssr()
            graph_normal_counter = Counter()
            for ring_size, count in graph_ring_counter.items():
                if count > graph_aromatic_counter[ring_size]:
                    graph_normal_counter[ring_size] = count - graph_aromatic_counter[ring_size]
            graph_total_ring = sum(graph_ring_counter.values())
            graph_total_aromatic = sum(graph_aromatic_counter.values())
            graph_total_normal = graph_total_ring - graph_total_aromatic
            # check normal rings
            if graph_total_normal > 0:
                max_ring_use = min(max_ring_use,
                                   int((total_ring_range[1]-total_aromatic_range[0])/graph_total_normal))
                if max_ring_use <= 0:
                    if logger.isEnabledFor(logging.INFO):
                        logger.info('ring_pattern: skip %s due to ring total %d', graph.to_smiles(), max_ring_use)
                    continue
            for ring_size, count in graph_normal_counter.items():
                if ring_size in ring_range:
                    if ring_size in aromatic_range:
                        max_ring_use = min(max_ring_use,
                                           int((ring_range[ring_size][1]-aromatic_range[ring_size][0])/count))
                    else:
                        max_ring_use = min(max_ring_use,
                                           int(ring_range[ring_size][1]/count))
                else:
                    max_ring_use = 0
                    break
            if max_ring_use <= 0:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('ring_pattern: skip %s due to ring %d', graph.to_smiles(), max_ring_use)
                continue
            # check aromatic rings
            if graph_total_aromatic > 0:
                max_ring_use = min(max_ring_use, int(total_aromatic_range[1]/graph_total_aromatic))
            for ring_size, count in graph_aromatic_counter.items():
                if ring_size in aromatic_range:
                    max_ring_use = min(max_ring_use, int(aromatic_range[ring_size][1]/count))
                else:
                    max_ring_use = 0
                    break
            if max_ring_use <= 0:
                if logger.isEnabledFor(logging.INFO):
                    logger.info('ring_pattern: skip %s due to aring %d', graph.to_smiles(), max_ring_use)
                continue
            # acceptable if max_use is positive
            good_rings[smiles] = max_ring_use
        return good_rings


# -----------------------------------------------------------------------------
# MoleculeEvaluator: evaluator for molecule structure in structure generation
# -----------------------------------------------------------------------------

class MoleculeEvaluator(object):
    """Class for evaluating molecular structure by regression models in structure generation

    Attributes:
        evaluator (FeatureEvaluator): evaluator of feature vector
        label (str): label component for the structure generation
        target_values (dict): a dictionary of property target values
        feature_dtype (np.array): data types of feature vector
        selection_mask (np.array): mask of feature selection
    """

    def __init__(self, evaluator, design_param, label, feature_vector_estimate=None):
        """Constructor of MoleculeEvaluator

        Args:
            evaluator (FeatureEvaluator): evaluator of feature vector
            design_param (DesignParam): design parameter
            label (str): label component for the structure generation
            feature_vector_estimate (array, optional): feature vector estimated in the feature estimation.
                Defaults to None.
        """
        self.evaluator = evaluator
        self.label = label
        self.target_values = design_param.get_target_values()
        self.feature_dtype = self.evaluator.get_feature_dtype(with_mask=False)
        self.selection_mask = self.evaluator.get_selection_mask()

        # set evaluator quiet mode
        self.evaluator.verbose = False

        if feature_vector_estimate is None:
            # make a feature vector
            self.feature_vector = np.zeros(shape=(sum(self.selection_mask)))
        else:
            self.feature_vector = feature_vector_estimate

    def is_linear_model(self):
        """Check if a regression model is linear

        Returns:
            bool: true if linear
        """
        return self.evaluator.is_linear_model()

    def get_fragments_in_feature_vector(self):
        """Get a set of fragments referred in evaluating a feature vector

        Returns:
            set: a set of fragment objects
        """
        referred_fragments = set()
        target_label = self.evaluator.get_target_label(self.label)
        index = 0
        for feature in target_label.get_feature_list():
            feature_size = feature.get_vector_size()
            if any(self.selection_mask[index:index+feature_size]):
                if isinstance(feature, FingerPrintStructureExtractor.Feature):
                    referred_fragments.add(feature.get_fragment())
            index += feature_size
        return referred_fragments

    def evaluate_molecule(self, node, local_vector=None, verify=False):
        """Evaluate a molecule represented by a search node

        Args:
            node (AtomGraphNode): a node of search tree
            local_vector (array, optional): a local feature vector. Defaults to None.
            verify (bool): a flag of verification of feature vector

        Returns:
            dict: a dictionary of property and estimated value
        """
        selected_feature_vector = np.array(self.feature_vector)
        target_label = self.evaluator.get_target_label(self.label)
        if local_vector is None:
            local_vector = self.get_local_feature_vector(node, verify)
            if local_vector is None:
                return None
        selection_slice = target_label.get_selection_slice()
        selected_feature_vector[selection_slice] = local_vector[target_label.get_selection_mask()]
        estimate = self.evaluator.estimate_property(self.target_values, selected_feature_vector)
        return estimate

    def get_local_feature_vector(self, node, verify=False):
        """Get a local feature vector from a search node

        Args:
            node (AtomGraphNode): a node of search tree
            verify (bool): a flag of verification of feature vector

        Returns:
            array: a local feature vector
        """
        target_label = self.evaluator.get_target_label(self.label)
        local_vector = np.zeros(len(target_label.get_selection_mask()))

        if verify:  # for debug
            graph = copy.deepcopy(node.graph)
            graph.expand_graph()
            smiles = graph.to_smiles()
            graph0 = AtomGraph(smiles=smiles)
            smiles0 = graph0.to_smiles()
            graph.reorder_canonical()
            labeling = ChemGraphLabeling(graph.vertices)
            atom_count = graph.get_atom_count()
            ring_count, aromatic_count = graph.count_rings_sssr()
            if smiles != smiles0:
                # smiles mismatch
                node_atom_list = []
                current_node = node
                while current_node.parent is not None:
                    node_atom_list.append(current_node.atom)
                    current_node = current_node.parent
                node_atom_list = reversed(node_atom_list)
                construction_path = ''
                for node_atom in node_atom_list:
                    construction_path += node_atom + ":"
                construction_path += '[' + smiles + ',' + smiles0 + ']'
                logger.error('smiles mismatch:{0}'.format(construction_path))

        index = 0
        valid_vector = True
        for features in target_label.get_features_list():
            for feature in features.get_feature_list():
                feature_size = feature.get_vector_size()
                if any(self.selection_mask[index:index+feature_size]):
                    # this feature is selected
                    if isinstance(feature, HeavyAtomExtractor.Feature):
                        value = node.atom_count[feature.get_index()]
                        local_vector[index] = value

                        if verify:  # for debug
                            ref = atom_count[feature.get_index()]
                            if value != ref:
                                logger.error('atom count mismatch:{0} {1} {2} {3}'.
                                             format(smiles, feature.get_id(), value, ref))

                    elif isinstance(feature, RingExtractor.Feature):
                        value = node.num_ring_count[feature.get_index()]
                        local_vector[index] = value

                        if verify:  # for debug
                            ref = ring_count[feature.get_index()]
                            if value != ref:
                                logger.error('ring count mismatch:{0} {1} {2} {3}'.
                                             format(smiles, feature.get_id(), value, ref))

                    elif isinstance(feature, AromaticRingExtractor.Feature):
                        value = node.num_aromatic_count[feature.get_index()]
                        local_vector[index] = value

                        if verify:  # for debug
                            ref = aromatic_count[feature.get_index()]
                            if value != ref:
                                logger.error('aromatic count mismatch:{0} {1} {2} {3}'.
                                             format(smiles, feature.get_id(), value, ref))

                    elif isinstance(feature, FingerPrintStructureExtractor.Feature):
                        value = node.fragment_count[feature.get_fragment()]
                        local_vector[index] = value

                        if verify:  # for debug
                            ref = feature.get_fragment().count_fragment_graph(graph, labeling)
                            if value != ref:
                                logger.error('fp structure mismatch:{0} {1} {2} {3}'.
                                             format(smiles+':'+smiles0, feature.get_id(), value, ref))

                    elif features.is_online_update():
                        value = node.feature_values[features.get_id()].get(feature.get_index(), None)
                        if value is not None:
                            features.copy_values(local_vector, index, feature, value)
                        else:
                            valid_vector = False
                            break

                    else:
                        valid_vector = False
                        logger.error('not-online feature in a feature vector: {0}'.format(features.get_id()))

                index += feature_size

        return local_vector if valid_vector else None

    def get_coefs(self):
        """Get coefficients, shift, target value, prediction error of linear regression models

        Returns:
            list, list, list, list: a list of coefficients, shifts, target values, and prediction errors
        """
        coefs = []
        consts = []
        targets = []
        errors = []
        coef_map = self.evaluator.get_coef()
        for prop, (coef, shift, std) in coef_map.items():
            coefs.append(coef)
            consts.append(shift)
            targets.append(self.target_values[prop])
            errors.append(std)
        return coefs, consts, targets, errors

    def get_resource_index(self):
        """Get indices of feature in a feature vector.

         Returns:
             dict, dict, dict, dict, dict: a dictionary of indices for atom, ring, aromatic ring,
                sub-structure, fp-structure
         """
        target_label = self.evaluator.get_target_label(self.label)
        target_slice = target_label.get_vector_slice()
        atom_index = dict()
        ring_index = dict()
        aromatic_index = dict()
        sub_fragment_index = dict()
        fp_fragment_index = dict()
        feature_index = dict()
        index = 0
        select_index = 0
        for features in target_label.get_features_list():
            for feature in features.get_feature_list():
                feature_size = feature.get_vector_size()
                if any(self.selection_mask[index:index+feature_size]):
                    # this feature is selected
                    if isinstance(feature, HeavyAtomExtractor.Feature):
                        atom_index[feature] = target_slice.start + select_index
                        select_index += 1
                    elif isinstance(feature, RingExtractor.Feature):
                        ring_index[feature] = target_slice.start + select_index
                        select_index += 1
                    elif isinstance(feature, AromaticRingExtractor.Feature):
                        aromatic_index[feature] = target_slice.start + select_index
                        select_index += 1
                    elif isinstance(feature, FingerPrintStructureExtractor.Feature):
                        fp_fragment_index[feature] = target_slice.start + select_index
                        select_index += 1
                    elif features.is_online_update():
                        for idx in range(feature_size):
                            if self.selection_mask[index+idx]:
                                feature_index[(feature, idx)] = target_slice.start + select_index
                                select_index += 1
                    else:
                        logger.error('unknown feature in a feature vector: {0}'.format(feature.get_id()))
                        select_index += sum(self.selection_mask[index:index+feature_size])
                index += feature_size
        return atom_index, ring_index, aromatic_index, sub_fragment_index, fp_fragment_index, feature_index
