# Enzyme Optimization Experiment

## Description
This script performs an optimization experiment for enzyme sequences using different mutation strategies.

## Import modules
```python
import logging
import pandas as pd
from gt4sd.frameworks.enzeptional.processing import HFandTAPEModelUtility
from gt4sd.frameworks.enzeptional.core import SequenceMutator, EnzymeOptimizer
from gt4sd.configuration import sync_algorithm_with_s3
from gt4sd.configuration import GT4SDConfiguration
configuration = GT4SDConfiguration.get_instance()
```

## Load datasets and scorers
```python
sync_algorithm_with_s3("proteins/enzeptional/scorers", module="properties")
```
Feasibility scorer path
```python
scorer_path = f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/feasibility/model.pkl"
```
## Set embedding model/tokenizer paths
```python
language_model_path = "facebook/esm2_t33_650M_UR50D"
tokenizer_path = "facebook/esm2_t33_650M_UR50D"
unmasking_model_path = "facebook/esm2_t33_650M_UR50D"
chem_model_path = "seyonec/ChemBERTa-zinc-base-v1"
chem_tokenizer_path = "seyonec/ChemBERTa-zinc-base-v1"
```
## Load protein embedding model
```python
protein_model = HFandTAPEModelUtility(
        embedding_model_path=language_model_path, tokenizer_path=tokenizer_path
    )
```
## Create mutation config 
```python
mutation_config = {
        "type": "language-modeling",
        "embedding_model_path": language_model_path,
        "tokenizer_path": tokenizer_path,
        "unmasking_model_path": unmasking_model_path,
    }
```
## Set key parameters
```python
intervals = [(5, 10), (20, 25)]
batch_size = 5
top_k = 3
substrate_smiles = "NC1=CC=C(N)C=C1"
product_smiles = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"

sample_sequence = "MSKLLMIGTGPVAIDQFLTRYEASCQAYKDMHQDQQLSSQFNTNLFEGDKALVTKFLEINRTLS"
```
## Load mutator
```python
mutator = SequenceMutator(sequence=sample_sequence, mutation_config=mutation_config)
```
## Set Optimizer
```python
optimizer = EnzymeOptimizer(
    sequence=sample_sequence,
    protein_model=protein_model,
    substrate_smiles=substrate_smiles,
    product_smiles=product_smiles,
    chem_model_path=chem_model_path,
    chem_tokenizer_path=chem_tokenizer_path,
    scorer_filepath=scorer_path,
    mutator=mutator,
    intervals=intervals,
    batch_size=batch_size,
    top_k=top_k,
    selection_ratio=0.25,
    perform_crossover=True,
    crossover_type="single_point",
    concat_order=["substrate", "sequence", "product"],
)
```
## Define optmization parameters
```python
num_iterations = 3
num_sequences = 5
num_mutations = 5
time_budget = 3600
```
## Optimize
```python
optimized_sequences, iteration_info = optimizer.optimize(
    num_iterations=num_iterations,
    num_sequences=num_sequences,
    num_mutations=num_mutations,
    time_budget=time_budget,
)
```