# -*- coding:utf-8 -*-
"""
molgxsdk.py

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

import os
from .Molecule import *
from .DataBox import *
from .FeatureExtraction import *
from .Prediction import *
from .FeatureEstimation import *
from .Generation import *
from .Utility import *

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MolgxSdk():
    def __init__(self, pickle_filepath = 'pickle/model.pkl'):
        self.pickle_filepath = os.path.abspath(pickle_filepath)

    def LoadPickle(self, pickle_filepath = None):
        target_property = {}

        if pickle_filepath is None:
            pickle_filepath = self.pickle_filepath

        if not os.path.exists(pickle_filepath):
            logger.error('{} does not exist'.format(pickle_filepath))
            return None
        else:
            moldata = MolData.load(pickle_filepath)

            tp_list = list(moldata.regression_model_list.keys())
            for t in tp_list:
                target_property[t] = ()

            return moldata, target_property

    def GenMols(self, moldata, para={}):
        # config
        try:
            prediction_error = para['prediction_error']
        except:
            prediction_error = 0.1

        # search
        try:
            without_estimate = para['without_estimate']
        except:
            without_estimate = False

        try:
            num_candidate = int(para['num_candidate'])
        except:
            num_candidate = 10

        try:
            max_molecule = int(para['max_molecule'])
        except:
            max_molecule = 100

        try:
            max_candidate = int(para['max_candidate'])
        except:
            max_candidate = 50

        try:
            max_solution = int(para['max_solution'])
        except:
            max_solution = 50

        try:
            max_node = int(para['max_node'])
        except:
            max_node = 200000

        try:
            beam_size = int(para['beam_size'])
        except:
            beam_size = 1000

        #
        target_property = para['target_property']

        #
        df_models = {}
        for target_p in target_property:
            df_m = moldata.get_regression_model_summary(target_p, models=[SklearnLinearRegressionModel])
            df_models[target_p] = df_m

        # get best model for each target property
        best_models = []
        for i,property in enumerate(target_property):
            df_model = df_models.get(property, None)
            if df_model is not None:
                best_model = df_model['model'][0]
                best_models.append(best_model)

        
        design_param = moldata.make_design_parameter(target_property,
                                                     prediction_error=prediction_error)

        # generation
        if without_estimate:
            molecules = moldata.generate_molecule(best_models,
                                                design_param,
                                                without_estimate=True,
                                                max_solution=max_solution,
                                                max_node=max_node,
                                                beam_size=beam_size)
        else:
            molecules = moldata.estimate_and_generate(best_models,
                                                    design_param,
                                                    num_candidate=num_candidate,
                                                    max_candidate=max_candidate,
                                                    max_molecule=max_molecule,
                                                    max_solution=max_solution,
                                                    max_node=max_node,
                                                    beam_size=beam_size)

        if len(molecules) > 0:
            df_molecule = moldata.get_generated_molecule_summary(best_models, design_param, mols=True, molobj=False, features=True)

        else:
            df_molecule = pd.DataFrame({})

        return df_molecule
