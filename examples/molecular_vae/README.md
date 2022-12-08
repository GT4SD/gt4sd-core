# Molecular VAE

### A short tutorial on training a molecular VAE to generate SMILES and subsequently use it for inference

#### Training 
This trains a molecular VAE using encoder and decoder based on StackGRUs. For details, see the PaccMannRL papers, for [omics](https://www.sciencedirect.com/science/article/pii/S2589004221002376) and [proteins](https://iopscience.iop.org/article/10.1088/2632-2153/abe808/meta).

```
conda activate gt4sd
gt4sd-trainer  --training_pipeline_name paccmann-vae-trainer --epochs 250 --batch_size 4 --n_layers 1 --rnn_cell_size 16 --latent_dim 16 --train_smiles_filepath src/gt4sd/training_pipelines/tests/molecules.smi --test_smiles_filepath src/gt4sd/training_pipelines/tests/molecules.smi --model_path /tmp/gt4sd-paccmann-gp/ --training_name fast-example --eval_interval 15 --save_interval 15 --selfies
```
*NOTE*: You might want to pass a SMILES/SELFIES language object via `--smiles_language_filepath`
*NOTE*: This is *dummy* example, do not use "as is" :warning:


#### Inference
This training pipeline trains a molecular VAE. The GT4SD **inference** pipeline support inference for PaccMannRL, i.e., a hybrid-VAE with an omics or a protein encoder and a molecular decoder.
There's not (yet) a dedicated inference pipeline for a normal molecular VAE, hence `gt4sd-saving` will not work (you could train the GuacaMol VAE instead via GT4SD).

To use a molecular VAE trained with `gt4sd-trainer  --training_pipeline_name paccmann-vae-trainer`, do the following:

``` sh
cd ~/.gt4sd/algorithms/conditional_generation/PaccMannRL/PaccMannRLProteinBasedGenerator/
cp -r v0 your_model_name # if this path does not exist, trigger the download via running the example from the README
cd your_model_name

# overwrite the weights file (.pt) with your desired checkpoint
cp path_to_your_model_file selfies_conditional_generator.pt  # use that filename even if you dont do SELFIES

# overwrite the selfies_language with the one from your model
cp path_to_your_smiles_language selfies_language.pkl
```
Afterwards all that is left, is to enter the `selfies_conditional_generator.json` and change the keys `vocab_size` and `embedding_size` to the size of your vocabulary.


Afterwards you can use as follows:
```py
import torch
from gt4sd.algorithms.conditional_generation.paccmann_rl.core import PaccMannRLProteinBasedGenerator, PaccMannRL

configuration = PaccMannRLProteinBasedGenerator(algorithm_version='your_model_name')
# Placeholder target (will not be used)
algorithm = PaccMannRL(configuration=configuration, target='')
model = configuration.get_conditional_generator(algorithm.local_artifacts)

# Set as desired
sample_size = 100
batch_size = 32

latent_size = 128  # dimensionality of the VAE latent code

smiles = []
while len(smiles) < sample_size:
    # Define latent code
    latent = torch.randn(1, batch_size, latent_size)

    # Bypass algorithm.sample by decoding SMILES directly from latent
    generated_smiles = model.get_smiles_from_latent(latent)
    _, valid_ids = model.validate_molecules(generated_smiles)
    generated_molecules = list([generated_smiles[index] for index in valid_ids])
    smiles.extend(generated_molecules)

print(smiles)
```
