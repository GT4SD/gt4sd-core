<!--
MIT License

Copyright (c) 2023 GT4SD team

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
-->

# Integrating Genetic Algorithms and Language Models for Enhanced Enzyme Design

## Overview
This study introduces a framework that combines LLMs with genetic algorithms (GAs) to optimize enzymes. LLMs are trained on a large dataset of protein sequences to learn the underlying statistical patterns and relationships. This knowledge is then leveraged by GAs to efficiently search for sequences with improved catalytic performance. It includes the capability to work with transition matrices for suggesting mutations.

## Requirements
- Python 3.6 or higher
- PyTorch
- Hugging Face's Transformers
- TAPE (Tasks Assessing Protein Embeddings)
- NumPy
- Joblib
- Logging module
- xgboost (optional)

## Installation
Ensure all required libraries are installed. You can install them using pip:
```bash
pip install torch transformers numpy joblib xgboost
```

## Usage
1. **Setting Global Cache Directories**: The script sets cache directories for Transformers and Torch.
2. **Model Cache**: Handles caching of models to avoid reloading.
3. **Utility Functions and Classes**:
    - `get_device`: Determines if CUDA is available for PyTorch.
    - `StringEmbedding`: An abstract class for embedding strings.
    - `HFandTAPEModelUtility`: Handles embeddings for both Hugging Face and TAPE models.
4. **Mutation Strategies**:
    - `MutationStrategy`: An abstract base class for mutation strategies.
    - `LanguageModelMutationStrategy`: Implements mutation using language models.
    - `TransitionMatrixMutationStrategy`: Uses a transition matrix for mutations.
5. **Sequence Mutator**: Generates mutations for protein sequences.
6. **Sequence Scorer**: Scores sequences using embeddings from protein and chemical models.
7. **Protein Sequence Optimizer**: Implements the genetic algorithm for protein sequence optimization. It handles the selection, crossover, and scoring of sequences over multiple iterations.

### Example Usage
```python
# Set Up Model Paths
language_model_path = "Rostlab/prot_bert"
tokenizer_path = "Rostlab/prot_bert"
unmasking_model_path = "Rostlab/prot_bert"
chem_model_path = "Rostlab/prot_bert" 
chem_tokenizer_path = "Rostlab/prot_bert"

protein_model = HFandTAPEModelUtility(embedding_model_path=language_model_path,
                                      tokenizer_path=tokenizer_path,)

# Mutation Configuration
mutation_config = {
    "type": "language-modeling",
    "embedding_model_path": language_model_path,
    "tokenizer_path": tokenizer_path,
    "unmasking_model_path": unmasking_model_path
}

# Define Parameters
intervals = [[5, 10], [20, 25]]
batch_size = 5
top_k = 3
substrate_smiles = "CCCO"  # Replace with actual substrate SMILES
product_smiles = "CCCO"  # Replace with actual product SMILES

# Initialize Sequence Mutator
sample_sequence = "WLSNIDMILRSPYSHTGAVLIYKQPDNNEDNIHPSSSMYFDANILIEALSKALVP"
mutator = SequenceMutator(sequence=sample_sequence, mutation_config=mutation_config)

# Initialize Protein Sequence Optimizer
optimizer = ProteinSequenceOptimizer(
    sequence=sample_sequence,
    protein_model=protein_model,
    substrate_smiles=substrate_smiles,
    product_smiles=product_smiles,
    chem_model_path=chem_model_path,
    chem_tokenizer_path=chem_tokenizer_path,
    mutator=mutator,
    intervals=intervals,
    batch_size=batch_size,
    top_k=top_k,
    selection_ratio=0.5,
    perform_crossover=True,
    crossover_type="single_point",
    concat_order=["substrate", "sequence", "product"]
)

# Run optimization
optimized_sequences, iteration_info = optimizer.optimize(
    num_iterations=5,
    num_sequences=50,
    num_mutations=5,
    time_budget=3600
)

# Output results
for i in optimized_sequences:
    seq = i["sequence"]
    score = i["score"]
    print(f"Sequence: {seq}, Score: {score}")

print(iteration_info)
```

## Customization
- Modify `intervals` to specify mutation regions in the sequence.
- Adjust `batch_size`, `top_k`, `selection_ratio`, and `crossover_type` for different optimization strategies.
- Change `concat_order` to alter the order of sequence, substrate, and product in the final embedding for scoring.
- Use `time_budget` to set a maximum time limit for each optimization iteration.

## Notes
- Ensure the paths to the models and tokenizers are correctly set.
- The script is designed for flexibility and can be adapted to different models and optimization strategies.
- For extensive usage, consider parallelizing or distributing the computation, especially for large-scale optimizations.

