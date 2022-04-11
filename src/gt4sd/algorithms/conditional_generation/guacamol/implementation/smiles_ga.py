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
"""SMILES GA implementation."""

from guacamol_baselines.smiles_ga.goal_directed_generation import ChemGEGenerator


class SMILESGA:
    def __init__(
        self,
        smi_file,
        population_size: int,
        n_mutations: int,
        n_jobs: int,
        random_start: bool,
        gene_size: int,
        generations: int,
        patience: int,
    ):
        """Initialize SMILESGA.

        Args:
            smi_file: path where to load hypothesis, candidate labels and, optionally, the smiles file.
            population_size: used with n_mutations for the initial generation of smiles within the population.
            n_mutations: used with population size for the initial generation of smiles within the population.
            n_jobs: number of concurrently running jobs.
            random_start: set to True to randomly choose list of SMILES for generating optimizied molecules.
            gene_size: size of the gene which is used in creation of genes.
            generations: number of evolutionary generations.
            patience: used for early stopping if population scores remains the same after generating molecules.
        """
        self.smi_file = smi_file
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.n_jobs = n_jobs
        self.random_start = random_start
        self.gene_size = gene_size
        self.generations = generations
        self.patience = patience

    def get_generator(self) -> ChemGEGenerator:
        """Create an instance of ChemGEGenerator.

        Returns:
            an instance of ChemGEGenerator.
        """
        optimiser = ChemGEGenerator(
            smi_file=self.smi_file,
            population_size=self.population_size,
            n_mutations=self.n_mutations,
            generations=self.generations,
            n_jobs=self.n_jobs,
            random_start=self.random_start,
            gene_size=self.gene_size,
            patience=self.patience,
        )
        return optimiser
