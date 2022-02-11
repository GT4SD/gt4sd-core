"""GRAPH GA implementation."""

from guacamol_baselines.graph_ga.goal_directed_generation import GB_GA_Generator


class GraphGA:
    def __init__(
        self,
        smi_file,
        mutation_rate: float,
        population_size: int,
        offspring_size: int,
        n_jobs: int,
        random_start: bool,
        generations: int,
        patience: int,
    ):
        """Initialize SMILESGA.

        Args:
            smi_file: path where to load hypothesis, candidate labels and, optionally, the smiles file.
            population_size: used with n_mutations for the initial generation of smiles within the population
            n_jobs: number of concurrently running jobs
            random_start: set to True to randomly choose list of SMILES for generating optimizied molecules
            generations: number of evolutionary generations
            patience: used for early stopping if population scores remains the same after generating molecules
            mutation_rate: frequency of the new mutations in a single gene or organism over time
            offspring_size: number of molecules to select for new population
        """
        self.smi_file = smi_file
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.n_jobs = n_jobs
        self.random_start = random_start
        self.generations = generations
        self.patience = patience

    def get_generator(self) -> GB_GA_Generator:
        """
        used for creating an instance of the GB_GA_Generator

        Returns:
            An instance of GB_GA_Generator
        """
        optimiser = GB_GA_Generator(
            smi_file=self.smi_file,
            population_size=self.population_size,
            offspring_size=self.offspring_size,
            mutation_rate=self.mutation_rate,
            generations=self.generations,
            n_jobs=self.n_jobs,
            random_start=self.random_start,
            patience=self.patience,
        )
        return optimiser
