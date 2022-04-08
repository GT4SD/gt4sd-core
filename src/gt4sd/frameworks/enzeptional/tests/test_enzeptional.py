# %%
import pandas as pd

from gt4sd.frameworks.enzeptional.core import Mutations
from gt4sd.frameworks.enzeptional.genetic_algorithm import (
    EnzymeDesignerGeneticAlgorithm,
)
from gt4sd.frameworks.enzeptional.processing import ProtTransXL

# %%
filepath = "/dccstor/yna/gt4sd-core/src/gt4sd/frameworks/enzeptional/tests/small_exmaple_ProtTrans_scorrer.pkl"

substrate = "NC1=CC=C(N)C=C1"
product = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"
sequence = "EGALFVEAESSHVLEDFGDFRPNDELHRVMVPTCDYSKGISSFPLLMVQLT"

mutation_path = "/dccstor/yna/gt4sd-core/src/gt4sd/frameworks/enzeptional/tests/mutation_scheme.json"

# %%

# Initialize the Enzyme Designer
designer = EnzymeDesignerGeneticAlgorithm(
    scorer_filepath=filepath,
    substrate=substrate,
    product=product,
    sequence=sequence,
    protein_embedding=ProtTransXL(),
)

# %%

# Optimize the Enzyme Designer
results = designer.optimize(
    number_of_mutations=15,
    number_of_steps=2,
    intervals=[(317, 350), (383, 393)],
    number_of_samples_per_step=5,
    mutations=Mutations.from_json(mutation_path),
)

# %%

result_path = "/PATH_TO_SAVE_RESULTS"
df = pd.DataFrame(results)
df.to_csv(result_path, encoding="utf-8", index=False)
# %%
