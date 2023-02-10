#
# MIT License
#
# Copyright (c) 2023 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import warnings

import pkg_resources

from gt4sd.frameworks.enzeptional import  core
from gt4sd.frameworks.enzeptional.processing import AutoModelFromHFEmbedding, TAPEEmbedding

warnings.simplefilter(action="ignore", category=FutureWarning)


filepath = pkg_resources.resource_filename(
    "gt4sd",
    "frameworks/enzeptional/tests/scorer.pkl",
)


substrate = "NC1=CC=C(N)C=C1"
product = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"
sequence = "EGALFVEAESSHVLEDFGDFRPNDELHRVMVPTCDYSKGISSFPLLMVQLTAESSHVLEDFGDFRPNVMVPTCDYSKGISSFPLLMVQLMVPTCDY"

designer = core.EnzymeOptimizer(
                    scorer_filepath=filepath, substrate=substrate, product=product, sequence=sequence,
                    protein_embedding=TAPEEmbedding()
                )

model_path = pkg_resources.resource_filename(
    "gt4sd",
    "frameworks/enzeptional/tests/mutation_model",
)
mutation_object = core.MutationLanguageModel(mutation_model_type="transformers", 
                                             mutation_model_parameters=
                                             {
                                                 "mutation_model_path":model_path, 
                                                 "mutation_tokenizer_filepath":model_path
                                                 }
                                             )


results = designer.optimize(
    number_of_mutations=2,
    intervals=[(41,43), (106,107), (145,149), (174,176), (182,184), (376,379)],
    number_of_steps=2,
    full_sequence_embedding = True,
    top_k=3,
    mutation_generator_type="language-modeling",
    batch_size=32,
    mutation_generator_parameters = {"maximum_number_of_mutations":5, "mutation_object": mutation_object},
    population_per_itaration = 20,
    with_genetic_algorithm=True,
    pad_intervals = True,
    top_k_selection=8,
    time_budget=2400
    )

