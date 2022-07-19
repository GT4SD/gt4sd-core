#
# MIT License
#
# Copyright (c) 2022 GT4SD team
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


filepath = pkg_resources.resource_filename(
    "gt4sd",
    "frameworks/enzeptional/tests/scorer.pkl",
)


substrate = "NC1=CC=C(N)C=C1"
product = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"
sequence = "EGALFVEAESSHVLEDFGDFRPNDELHRVMVPTCDYSKGISSFPLLMVQLTAESSHVLEDFGDFRPNVMVPTCDYSKGISSFPLLMVQLMVPTCDY"

designer = core.EnzymeOptimizer(
    scorer_filepath=filepath,
    substrate=substrate,
    product=product,
    sequence=sequence,
    protein_embedding=core.Embedder(
        "Rostlab/prot_t5_xl_uniref50", "Rostlab/prot_t5_xl_uniref50"
    ),
)

mutation_model = pkg_resources.resource_filename(
    "gt4sd",
    "frameworks/enzeptional/tests/mutation_model",
)

mutation_model_tokenizer = pkg_resources.resource_filename(
    "gt4sd",
    "frameworks/enzeptional/tests/mutation_model/tokenizer.json",
)

mutation_object = core.MutationLanguageModel(mutation_model, mutation_model_tokenizer)

results = designer.optimize(
    number_of_mutations=1,
    intervals=[(5, 19), (25, 40)],
    number_of_steps=2,
    full_sequence_embedding=True,
    top_k=1,
    mutation_generator_type="language-modeling",
    batch_size=2,
    mutation_generator_parameters={
        "maximum_number_of_mutations": 4,
        "mutation_object": mutation_object,
    },
    population_per_itaration=5,
    with_genetic_algorithm=True,
    top_k_selection=1,
    selection_method="generic",
    crossover_method="single_point",
)
