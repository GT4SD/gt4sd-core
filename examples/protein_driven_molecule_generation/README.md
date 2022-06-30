# Protein-driven molecule generation

You can use the **PaccMann<sup>GP</sup>** model, a VAE pretrained on drug-like compounds from ChEMBL, to generate molecules with high binding affinity to a desired protein.
**PaccMann<sup>GP</sup>** relies on Gaussian processes to navigate a static latent space of the pretrained VAE.

Say, you are interested in discovering potential inhibitors of the [JAK1 kinase (UniProt ID: P23458)](https://www.uniprot.org/uniprot/P23458). Then, leverage **PaccMann<sup>GP</sup>** as follows:

```py
from gt4sd.algorithms.controlled_sampling.paccmann_gp import PaccMannGPGenerator, PaccMannGP
configuration = PaccMannGPGenerator()
target = {
    "affinity": {"protein": "LGEGHFGKVAKELVLMEFLPSGERNLGDL"}
}
paccmann_gp = PaccMannGP(configuration=configuration, target=target)
items = list(paccmann_gp.sample(10))
```

If you were simultaneously interested to optimize other properties like QED or SCScore, define this in your target specification.
You can also add weights to each component of the multi-objective. 

```py
target = {
    "qed": {"weight": 1.0},
    "sa": {"weight": 2.0},
    "affinity": {"protein": "LGEGHFGKVAKELVLMEFLPSGERNLGDL"}
}
```

For details on this methodology see:

```bib
@article{born2022active,
	author = {Born, Jannis and Huynh, Tien and Stroobants, Astrid and Cornell, Wendy D. and Manica, Matteo},
	title = {Active Site Sequence Representations of Human Kinases Outperform Full Sequence Representations for Affinity Prediction and Inhibitor Generation: 3D Effects in a 1D Model},
	journal = {Journal of Chemical Information and Modeling},
	volume = {62},
	number = {2},
	pages = {240-257},
	year = {2022},
	doi = {10.1021/acs.jcim.1c00889},
	note ={PMID: 34905358},
	URL = {https://doi.org/10.1021/acs.jcim.1c00889}
}
```
