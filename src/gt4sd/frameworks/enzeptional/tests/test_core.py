#
# MIT License
#
# Copyright (c) 2024 GT4SD team
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
from gt4sd.frameworks.enzeptional.core import (
    SequenceMutator,
    EnzymeOptimizer,
)

from gt4sd.frameworks.enzeptional.processing import HFandTAPEModelUtility

from gt4sd.configuration import sync_algorithm_with_s3
from gt4sd.configuration import GT4SDConfiguration

configuration = GT4SDConfiguration.get_instance()


warnings.simplefilter(action="ignore", category=FutureWarning)

sync_algorithm_with_s3("proteins/enzeptional/scorers", module="properties")

scorer_filepath = f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/feasibility/model.pkl"


def test_optimize():
    language_model_path = "facebook/esm2_t33_650M_UR50D"
    tokenizer_path = "facebook/esm2_t33_650M_UR50D"
    unmasking_model_path = "facebook/esm2_t33_650M_UR50D"
    chem_model_path = "seyonec/ChemBERTa-zinc-base-v1"
    chem_tokenizer_path = "seyonec/ChemBERTa-zinc-base-v1"

    protein_model = HFandTAPEModelUtility(
        embedding_model_path=language_model_path, tokenizer_path=tokenizer_path
    )

    mutation_config = {
        "type": "language-modeling",
        "embedding_model_path": language_model_path,
        "tokenizer_path": tokenizer_path,
        "unmasking_model_path": unmasking_model_path,
    }

    intervals = [(5, 10), (20, 25)]
    batch_size = 5
    top_k = 3
    substrate_smiles = "NC1=CC=C(N)C=C1"
    product_smiles = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"

    sample_sequence = "MSKLLMIGTGPVAIDQFLTRYEASCQAYKDMHQDQQLSSQFNTNLFEGDKALVTKFLEINRTLS"
    mutator = SequenceMutator(sequence=sample_sequence, mutation_config=mutation_config)

    optimizer = EnzymeOptimizer(
        sequence=sample_sequence,
        protein_model=protein_model,
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles,
        chem_model_path=chem_model_path,
        chem_tokenizer_path=chem_tokenizer_path,
        scorer_filepath=scorer_filepath,
        mutator=mutator,
        intervals=intervals,
        batch_size=batch_size,
        top_k=top_k,
        selection_ratio=0.25,
        perform_crossover=True,
        crossover_type="single_point",
        concat_order=["substrate", "sequence", "product"],
    )

    num_iterations = 3
    num_sequences = 5
    num_mutations = 5
    time_budget = 50000

    optimized_sequences, iteration_info = optimizer.optimize(
        num_iterations=num_iterations,
        num_sequences=num_sequences,
        num_mutations=num_mutations,
        time_budget=time_budget,
    )

    assert len(optimized_sequences) > 0
