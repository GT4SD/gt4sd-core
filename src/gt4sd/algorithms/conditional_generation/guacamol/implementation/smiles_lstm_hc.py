"""Recurrent Neural Networks with Hill Climbing algorithm implementation."""

from guacamol_baselines.smiles_lstm_hc.goal_directed_generation import (
    SmilesRnnDirectedGenerator,
)


class SMILESLSTMHC:
    def __init__(
        self,
        model_path: str,
        smi_file,
        max_len: int,
        n_jobs: int,
        keep_top: int,
        n_epochs: int,
        mols_to_sample: int,
        optimize_n_epochs: int,
        benchmark_num_samples: int,
        optimize_batch_size: int,
        random_start: bool,
    ):
        """Initialize SMILESLSTMHC.
        Args:
            model_path: path to load the model,
            smi_file: path to load the hypothesis, candidate labels and, optionally, the smiles file,
            max_len: maximum length of a SMILES string,
            n_jobs: number of concurrently running jobs,
            keep_top: molecules kept each step,
            n_epochs: number of epochs to sample,
            mols_to_sample: molecules sampled at each step,
            optimize_n_epochs: number of epochs for the optimization,
            benchmark_num_samples: number of molecules to generate from final model for the benchmark,
            optimize_batch_size: batch size for the optimization,
            random_start: set to True to randomly choose list of SMILES for generating optimizied molecules
            ,
        """
        self.model_path = model_path
        self.n_epochs = n_epochs
        self.mols_to_sample = mols_to_sample
        self.keep_top = keep_top
        self.optimize_n_epochs = optimize_n_epochs
        self.max_len = max_len
        self.optimize_batch_size = optimize_batch_size
        self.benchmark_num_samples = benchmark_num_samples
        self.random_start = random_start
        self.smi_file = smi_file
        self.n_jobs = n_jobs

    def get_generator(self) -> SmilesRnnDirectedGenerator:
        """
        used for creating an instance of the SmilesRnnDirectedGenerator

        Returns:
            An instance of SmilesRnnDirectedGenerator
        """
        optimiser = SmilesRnnDirectedGenerator(
            pretrained_model_path=self.model_path,
            n_epochs=self.n_epochs,
            mols_to_sample=self.mols_to_sample,
            keep_top=self.keep_top,
            optimize_n_epochs=self.optimize_n_epochs,
            max_len=self.max_len,
            optimize_batch_size=self.optimize_batch_size,
            number_final_samples=self.benchmark_num_samples,
            random_start=self.random_start,
            smi_file=self.smi_file,
            n_jobs=self.n_jobs,
        )
        return optimiser
