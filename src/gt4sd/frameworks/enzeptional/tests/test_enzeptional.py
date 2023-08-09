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
from gt4sd.frameworks.enzeptional import core

warnings.simplefilter(action="ignore", category=FutureWarning)

# Paths and data
scorer_filepath = pkg_resources.resource_filename(
    "gt4sd", "frameworks/enzeptional/tests/scorer.pkl"
)
substrate = "NC1=CC=C(N)C=C1"
product = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"
sequence = "EGALFVEAESSHVLEDFGDFRPNDELHRVMVPTCDYSKGISSFPLLMVQLTAESSHVLEDFGDFRPNVMVPTCDYSKGISSFPLLMVQLMVPTCDY"

# Enzyme optimizer setup
enzyme_optimizer = core.EnzymeOptimizer(scorer_filepath, substrate, product, sequence)

# Mutation language model setup
mutation_object = core.MutationLanguageModel(
    mutation_model_parameters={
        "model_path": "facebook/esm2_t33_650M_UR50D",
        "tokenizer_path": "facebook/esm2_t33_650M_UR50D",
    }
)

# Optimization
results = enzyme_optimizer.optimize(
    number_of_mutations=2,
    intervals=[(41, 43), (106, 107), (145, 149), (174, 176), (182, 184), (376, 379)],
    number_of_steps=2,
    full_sequence_embedding=True,
    top_k=3,
    mutation_generator="language-modeling",
    batch_size=4,
    mutation_generator_parameters={
        "maximum_number_of_mutations": 5,
        "mutation_object": mutation_object,
    },
    population_per_iteration=5,
    with_genetic_algorithm=True,
    pad_intervals=True,
    top_k_selection=2,
    time_budget=300,
)
