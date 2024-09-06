import logging
import pandas as pd
from typing import Tuple, List, Optional
from gt4sd.frameworks.enzeptional import (
    EnzymeOptimizer,
    SequenceMutator,
    SequenceScorer,
    CrossoverGenerator,
    HuggingFaceEmbedder,
    HuggingFaceModelLoader,
    HuggingFaceTokenizerLoader,
    SelectionGenerator,
)
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
    scorer =  f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/{model}/model.pkl"
    if model == "feasibility":
        return scorer, None
    else:
        scaler = f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/{model}/scaler.pkl"
        return scorer, scaler

def load_experiment_parameters(model="feasibility") -> Tuple[List, List, List, List]:
    """Load experiment parameters from a CSV file."""
    substrate_smiles = "NC1=CC=C(N)C=C1"
    product_smiles = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"
    intervals = [(5, 10), (20, 25)]
    sample_sequence = "MSKLLMIGTGPVAIDQFLTRYEASCQAYKDMHQDQQLSSQFNTNLFEGDKALVTKFLEINRTLS"
    scorer_path, scaler_path = initialize_environment(model)
    return substrate_smiles, product_smiles, sample_sequence, intervals, scorer_path, scaler_path


def setup_optimizer(
    substrate_smiles: str,
    product_smiles: str,
    sample_sequence: str,
    scorer_path: str,
    scaler_path: str,
    intervals: List[List[int]],
    concat_order: List[str],
    top_k: int,
    batch_size: int,
    use_xgboost_scorer: bool
):
    """Set up and return the optimizer with all necessary components configured

    Args:
        substrate_smiles (str): SMILES representation of
        the substrate.
        product_smiles (str): SMILES representation of the
        product.
        sample_sequence (str): The initial protein sequence.
        scorer_path (str): File path to the scoring model.
        scaler_path (str): Path to the scaller in case you are usinh the Kcat model.
        intervals (List[List[int]]): Intervals for mutation.
        concat_order (List[str]): Order of concatenating embeddings.
        top_k (int): Number of top amino acids to use to create mutants.
        batch_size (int): Batch size.
        use_xgboost_scorer (bool): flag to specify if the fitness function is the Kcat.

    Returns:
        Initialized optmizer
    """
    language_model_path = "facebook/esm2_t33_650M_UR50D"
    tokenizer_path = "facebook/esm2_t33_650M_UR50D"
    chem_model_path = "seyonec/ChemBERTa-zinc-base-v1"
    chem_tokenizer_path = "seyonec/ChemBERTa-zinc-base-v1"

    model_loader = HuggingFaceModelLoader()
    tokenizer_loader = HuggingFaceTokenizerLoader()

    protein_model = HuggingFaceEmbedder(
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        model_path=language_model_path,
        tokenizer_path=tokenizer_path,
        cache_dir=None,
        device="cpu",
    )

    chem_model = HuggingFaceEmbedder(
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        model_path=chem_model_path,
        tokenizer_path=chem_tokenizer_path,
        cache_dir=None,
        device="cpu",
    )

    mutation_config = {
        "type": "language-modeling",
        "embedding_model_path": language_model_path,
        "tokenizer_path": tokenizer_path,
        "unmasking_model_path": language_model_path,
    }

    mutator = SequenceMutator(sequence=sample_sequence, mutation_config=mutation_config)
    mutator.set_top_k(top_k)

    scorer = SequenceScorer(
        protein_model=protein_model,
        scorer_filepath=scorer_path,
        use_xgboost=use_xgboost_scorer,
        scaler_filepath=scaler_path,
    )

    selection_generator = SelectionGenerator()
    crossover_generator = CrossoverGenerator()
    
    optimizer_config = dict(
        sequence=sample_sequence,
        mutator=mutator,
        scorer=scorer,
        intervals=intervals,
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles,
        chem_model=chem_model,
        selection_generator=selection_generator,
        crossover_generator=crossover_generator,
        concat_order=concat_order,
        batch_size=batch_size,
        selection_ratio=0.25,
        perform_crossover=True,
        crossover_type="single_point",
        pad_intervals=False,
        minimum_interval_length=8,
        seed=42,
    )
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
    concat_order = ["substrate", "sequence"]
    use_xgboost_scorer=True
    top_k=2
    batch_size=2
    substrate_smiles, product_smiles, sample_sequence, intervals, scorer_path, scaler_path = load_experiment_parameters("kcat")
    optimizer = setup_optimizer(
        substrate_smiles,
        product_smiles,
        sample_sequence,
        scorer_path,
        scaler_path,
        intervals,
        concat_order,
        top_k,
        batch_size,
        use_xgboost_scorer
    )

    optimized_sequences, iteration_info = optimize_sequences(optimizer)
    logging.info("Optimization completed.")


def main_feasibility():
    """Optimization using Feasibility model"""    
    logging.basicConfig(level=logging.INFO)
    concat_order = ["substrate", "sequence", "product"]
    use_xgboost_scorer=False
    top_k=2
    batch_size=2
    substrate_smiles, product_smiles, sample_sequence, intervals, scorer_path, scaler_path = load_experiment_parameters("feasilibity")
    optimizer = setup_optimizer(
        substrate_smiles,
        product_smiles,
        sample_sequence,
        scorer_path,
        scaler_path,
        intervals,
        concat_order,
        top_k,
        batch_size,
        use_xgboost_scorer
    )

    optimized_sequences, iteration_info = optimize_sequences(optimizer)
    logging.info("Optimization completed.")

if __name__ == "__main__":
    main_feasibility()
    main_kcat()
