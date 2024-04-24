import logging
import pandas as pd
from typing import Tuple, List, Optional
from gt4sd.frameworks.enzeptional.processing import HFandTAPEModelUtility
from gt4sd.frameworks.enzeptional.core import SequenceMutator, EnzymeOptimizer
from gt4sd.configuration import GT4SDConfiguration, sync_algorithm_with_s3


def initialize_environment(model = "feasibility") -> Tuple[str, Optional[str]]:
    """Synchronize with GT4SD S3 storage and set up the environment.
    
    Args:
        model (str): Type of optimization ("feasibility" or "kcat").

    Returns:
        Tuple[str, Optional[str]]: The path to the scorer file and scaler file (if existing). 
    """    
    configuration = GT4SDConfiguration.get_instance()
    sync_algorithm_with_s3("proteins/enzeptional/scorers", module="properties")
    name = model.lower()
    if name == "kcat":
        return f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/{name}/model.pkl", f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/{name}/scaler.pkl"
    else:
        return f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/{name}/model.pkl", None


def load_experiment_parameters() -> Tuple[List, List, List, List]:
    """Load experiment parameters from a CSV file."""
    df = pd.read_csv("data.csv").iloc[1]
    return df["substrates"], df["products"], df["sequences"], eval(df["intervals"])


def setup_optimizer(
    substrate_smiles: str,
    product_smiles: str,
    sample_sequence: str,
    intervals: List[List[int]],
    scorer_path: str,
    scaler_path: str,
    concat_order: List[str],
    use_xgboost_scorer: bool
):
    """Set up and return the optimizer with all necessary components configured

    Args:
        substrate_smiles (str): SMILES representation of
        the substrate.
        product_smiles (str): SMILES representation of the
        product.
        sample_sequence (str): The initial protein sequence.
        intervals (List[List[int]]): Intervals for mutation.
        scorer_path (str): File path to the scoring model.
        scaler_path (str): Path to the scaller in case you are usinh the Kcat model.
        concat_order (List[str]): Order of concatenating embeddings.
        use_xgboost_scorer (bool): flag to specify if the fitness function is the Kcat.

    Returns:
        Initialized optmizer
    """
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
        "concat_order": concat_order,
        "scaler_filepath": scaler_path,
        "use_xgboost_scorer": use_xgboost_scorer
    }
    return EnzymeOptimizer(**optimizer_config)


def optimize_sequences(optimizer):
    """Optimize sequences using the configured optimizer.

    Args:
        optimizer: Initialized optimizer

    Returns:
        Optimized sequences
    """    
    return optimizer.optimize(
        num_iterations=3, num_sequences=5, num_mutations=5, time_budget=3600
    )


def main_kcat():
    """Optimization using Kcat model"""    
    logging.basicConfig(level=logging.INFO)
    scorer_path, scaler_path = initialize_environment(model="kcat")
    concat_order, use_xgboost_scorer = ["substrate", "sequence"], True
    (
        substrate_smiles,
        product_smiles,
        sample_sequence,
        intervals,
    ) = load_experiment_parameters()
    optimizer = setup_optimizer(
        substrate_smiles, product_smiles, sample_sequence, intervals, scorer_path, scaler_path, concat_order, use_xgboost_scorer
    )
    optimized_sequences, iteration_info = optimize_sequences(optimizer)
    logging.info("Optimization completed.")


def main_feasibility():
    """Optimization using Feasibility model"""    
    logging.basicConfig(level=logging.INFO)
    scorer_path, scaler_path = initialize_environment()
    concat_order, use_xgboost_scorer = ["substrate", "sequence", "product"], False
    (
        substrate_smiles,
        product_smiles,
        sample_sequence,
        intervals,
    ) = load_experiment_parameters()
    optimizer = setup_optimizer(
        substrate_smiles, product_smiles, sample_sequence, intervals, scorer_path, scaler_path, concat_order, use_xgboost_scorer
    )
    optimized_sequences, iteration_info = optimize_sequences(optimizer)
    logging.info("Optimization completed.")

if __name__ == "__main__":
    main_feasibility()
    main_kcat()
