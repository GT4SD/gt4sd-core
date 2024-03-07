import logging
import pandas as pd
from gt4sd.frameworks.enzeptional.processing import HFandTAPEModelUtility
from gt4sd.frameworks.enzeptional.core import SequenceMutator, EnzymeOptimizer
from gt4sd.configuration import GT4SDConfiguration, sync_algorithm_with_s3


def initialize_environment():
    """Synchronize with GT4SD S3 storage and set up the environment."""
    # NOTE: For those interested in optimizing kcat values, it is important to adjust the scorer path to reflect this focus, thereby selecting the appropriate model for kcat optimization: f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/kcat/model.pkl". The specification of the scaler, located within the same directory as the `scorer.pkl`, is mandatory for accurate model performance.
    configuration = GT4SDConfiguration.get_instance()
    sync_algorithm_with_s3("proteins/enzeptional/scorers", module="properties")
    return f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/feasibility/model.pkl"


def load_experiment_parameters():
    """Load experiment parameters from a CSV file."""
    df = pd.read_csv("data.csv").iloc[1]
    return df["substrates"], df["products"], df["sequences"], eval(df["intervals"])


def setup_optimizer(
    substrate_smiles, product_smiles, sample_sequence, intervals, scorer_path
):
    """Set up and return the optimizer with all necessary components configured."""
    model_tokenizer_paths = "facebook/esm2_t33_650M_UR50D"
    chem_paths = "seyonec/ChemBERTa-zinc-base-v1"

    protein_model = HFandTAPEModelUtility(
        embedding_model_path=model_tokenizer_paths, tokenizer_path=model_tokenizer_paths
    )
    mutation_config = {
        "type": "language-modeling",
        "embedding_model_path": model_tokenizer_paths,
        "tokenizer_path": model_tokenizer_paths,
        "unmasking_model_path": model_tokenizer_paths,
    }

    mutator = SequenceMutator(sequence=sample_sequence, mutation_config=mutation_config)
    optimizer_config = {
        "sequence": sample_sequence,
        "protein_model": protein_model,
        "substrate_smiles": substrate_smiles,
        "product_smiles": product_smiles,
        "chem_model_path": chem_paths,
        "chem_tokenizer_path": chem_paths,
        "scorer_filepath": scorer_path,
        "mutator": mutator,
        "intervals": intervals,
        "batch_size": 5,
        "top_k": 3,
        "selection_ratio": 0.25,
        "perform_crossover": True,
        "crossover_type": "single_point",
        "concat_order": ["substrate", "sequence", "product"],
    }
    return EnzymeOptimizer(**optimizer_config)


def optimize_sequences(optimizer):
    """Optimize sequences using the configured optimizer."""
    return optimizer.optimize(
        num_iterations=3, num_sequences=5, num_mutations=5, time_budget=3600
    )


def main():
    logging.basicConfig(level=logging.INFO)
    scorer_path = initialize_environment()
    (
        substrate_smiles,
        product_smiles,
        sample_sequence,
        intervals,
    ) = load_experiment_parameters()
    optimizer = setup_optimizer(
        substrate_smiles, product_smiles, sample_sequence, intervals, scorer_path
    )
    optimized_sequences, iteration_info = optimize_sequences(optimizer)
    logging.info("Optimization completed.")


if __name__ == "__main__":
    main()
