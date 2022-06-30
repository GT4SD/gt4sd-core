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

import pandas as pd

# from gt4sd.frameworks.enzeptional.genetic_algorithm import (
#     EnzymeOptimizerGeneticAlgorithm,
# )
from gt4sd.frameworks.enzeptional import genetic_algorithm_LM

warnings.simplefilter(action="ignore", category=FutureWarning)


filepath = "/Users/yna/PhD/enzeptional/scorers/short_seq_tape.pkl"

substrate = "NC1=CC=C(N)C=C1"
product = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"
sequence = "EGALFVEAESSHVLEDFGDFRPNDELHRVMVPTCDYSKGISSFPLLMVQLT"

mutation_path = "/Users/yna/PhD/enzeptional/gt4sd-core/src/gt4sd/frameworks/enzeptional/tests/mutation_scheme.json"



# Initialize the Enzyme Designer
# designer = core.EnzymeOptimizer(
#     scorer_filepath=filepath,
#     substrate=substrate,
#     product=product,
#     sequence=sequence,
#     protein_embedding_type="prottrans",
#     protein_embedding_path="prottrans",
# )

designer = genetic_algorithm_LM.EnzymeOptimizerGeneticAlgorithm(
    scorer_filepath=filepath,
    substrate=substrate,
    product=product,
    sequence=sequence,
    protein_embedding_path="tape",
    protein_embedding_type="tape",
    mutation_model_type="albert",
    mutation_model_path="/Users/yna/PhD/enzeptional/models/albert_from_scratch",
)




# Optimize the Enzyme Designer
# results = designer.optimize(
#     number_of_mutations=2,
#     number_of_steps=1,
#     intervals=[(10, 15), (18, 24)],
#     number_of_samples_per_step=2,
#     mutations=Mutations.from_json(mutation_path),
# )

results = designer.optimize(
    number_of_mutations=2,
    number_of_steps=1,
    intervals=[(10, 15), (18, 24)],
    pad_intervals=True,
    population_per_itaration=5,
    number_selection_per_iteration=2,
)



result_path = "/PATH_TO_SAVE_RESULTS"
df = pd.DataFrame(results)
print(df)
# df.to_csv(result_path, encoding="utf-8", index=False)

