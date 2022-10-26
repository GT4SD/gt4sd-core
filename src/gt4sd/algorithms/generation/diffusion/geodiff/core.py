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
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from rdkit import Chem
from diffusers import DDPMScheduler
from IPython.display import SVG, display
from nglview import show_rdkit as show
from rdkit.Chem.Draw import rdMolDraw2D as MD2
from torch_geometric.data import Data
from torch_scatter import scatter_mean

from .model.core import MoleculeGNN
from .model.utils import repeat_data, set_rdmol_positions

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GeoDiffPipeline:
    """Pipeline for molecular conformation generation using GeoDiff.
    The pipeline defined here is slightly different than the pipeline used in diffusers.

    GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation, Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, Jian Tang - https://arxiv.org/abs/2203.02923
    """

    def __init__(
        self, model_name_or_path: str, params_json: Optional[str] = None
    ) -> None:
        """GeoDiff pipeline for molecular conformation generation. Code adapted from colab:
                https://colab.research.google.com/drive/1pLYYWQhdLuv1q-JtEHGZybxp2RBF8gPs#scrollTo=-3-P4w5sXkRU written by Nathan Lambert.

        Args:
            model_name_or_path: pretrained model name or path to model directory.
            params_json: parameters as a JSON file. Defaults to None, a.k.a., use default configuration.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = model_name_or_path

        config = {}
        if params_json is not None:
            with open(params_json, "r") as f:
                config = json.load(f)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=getattr(config, "num_timesteps", 1000),
            beta_schedule=getattr(config, "beta_schedule", "linear"),
            beta_start=getattr(config, "beta_start", 1e-7),
            beta_end=getattr(config, "beta_end", 2e-3),
            clip_sample=getattr(config, "clip_sample", False),
        )
        sigmas = (
            torch.tensor(1.0 - self.scheduler.alphas_cumprod).sqrt()
            / torch.tensor(self.scheduler.alphas_cumprod).sqrt()
        )
        self.sigmas = sigmas.to(self.device)

        self.num_samples = 1  # solutions per molecule
        self.num_molecules = 3
        self.w_global = 0.5  # 0,.3 for qm9
        self.global_start_sigma = 0.5
        self.eta = 1.0
        self.clip_local = None
        self.clip_pos = None

        # constants for data handling
        self.save_traj = False
        self.save_data = False
        self.output_dir = "out"
        os.makedirs(self.output_dir, exist_ok=True)

        # load pretrained model
        self.model = MoleculeGNN.from_pretrained(model_name_or_path)

    def to(self, device: str = "cuda") -> None:
        """Move model to a device.

        Args:
            device: device where to move the model. Defaults to "cuda".
        """
        self.model.to(self.device)

    @classmethod
    def from_pretrained(
        self, model_name_or_path: str, params_json: Optional[str] = None
    ) -> "GeoDiffPipeline":
        """Load pretrained model.

        Args:
            model_name_or_path: pretrained model name or path to model directory.
            params_json: path to model config.

        Returns:
            a GeoDiff pipeline.
        """
        return GeoDiffPipeline(
            model_name_or_path=model_name_or_path, params_json=params_json
        )

    @torch.no_grad()
    def __call__(
        self, batch_size: int, prompt: Dict[str, Any]
    ) -> Dict[str, List[Chem.Mol]]:
        """Generate conformations for a molecule.

        Args:
            batch_size: number of samples to generate.
            prompt: `torch_geometric.data.Data` object containing the molecular graph in 2D format. This information is given as conditioning for the model.

        Returns:
            a dict containing a list of postprocessed generated conformations.
        """

        results = []
        # 2d representation for the molecule
        # convert dict in torch_geometric.data.Data
        data = Data.from_dict(prompt)
        num_samples = max(data.pos_ref.size(0) // data.num_nodes, 1)

        data_input = data.clone()
        data_input["pos_ref"] = None
        batch = repeat_data(data_input, num_samples).to(self.device)

        # initial configuration
        pos_init = torch.randn(batch.num_nodes, 3).to(self.device)

        # for logging animation of denoising
        pos_traj = []

        # sample from pretrained diffusion process
        with torch.no_grad():
            # scale initial sample
            pos = pos_init * self.sigmas[-1]
            for t in self.scheduler.timesteps:
                batch.pos = pos

                # generate geometry with model, then filter it
                epsilon = self.model.forward(
                    batch, t, sigma=self.sigmas[t], return_dict=False
                )[0]

                # Update
                reconstructed_pos = self.scheduler.step(epsilon, t, pos)[
                    "prev_sample"
                ].to(self.device)

                pos = reconstructed_pos

                if torch.isnan(pos).any():
                    raise FloatingPointError("NaN detected. Please restart.")

                # recenter graph of positions for next iteration
                pos = pos - scatter_mean(pos, batch.batch, dim=0)[batch.batch]

                # optional clipping
                if self.clip_pos is not None:
                    pos = torch.clamp(pos, min=-self.clip_pos, max=self.clip_pos)
                pos_traj.append(pos.clone().cpu())

        pos_gen = pos.cpu()

        if self.save_traj:
            pos_gen_traj = [pt.cpu() for pt in pos_traj]
            data.pos_gen = torch.stack(pos_gen_traj)
        else:
            data.pos_gen = pos_gen

        results.append(data)

        if self.save_data:
            save_path = os.path.join(self.output_dir, "samples.pkl")

            with open(save_path, "wb") as f:
                pickle.dump(results, f)

        mols_gen, mols_orig = self.postprocess_output(results)
        return {"sample": mols_gen}

    def postprocess_output(
        self, results: List[Data]
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        """Postprocess output of diffusion pipeline.

        Args:
            results: list of `torch_geometric.data.Data` objects containing the molecular graph in 3D format.

        Returns:
            tuple with list of postprocessed generated conformations and list of postprocessed original conformations.
        """

        # the model can generate multiple conformations per 2d geometry
        # num_gen = results[0]["pos_gen"].shape[0]
        # init storage objects
        mols_gen = []
        mols_orig = []
        for to_process in results:

            # store the reference 3d position
            to_process["pos_ref"] = to_process["pos_ref"].reshape(
                -1, to_process["rdmol"].GetNumAtoms(), 3
            )

            # store the generated 3d position
            to_process["pos_gen"] = to_process["pos_gen"].reshape(
                -1, to_process["rdmol"].GetNumAtoms(), 3
            )

            # copy data to new object
            new_mol = set_rdmol_positions(to_process.rdmol, to_process["pos_gen"][0])

            # append results
            mols_gen.append(new_mol)
            mols_orig.append(to_process.rdmol)

        logger.info(f"collect {len(mols_gen)} generated molecules in `mols`")
        return mols_gen, mols_orig

    def visualize_2d_input(self, data: Data) -> None:
        """Visualize 2D input.

        Args:
            data: `torch_geometric.data.Data` object containing the molecular graph in 2D format.
        """

        mc = Chem.MolFromSmiles(data[0]["smiles"])
        molSize = (450, 300)
        drawer = MD2.MolDraw2DSVG(molSize[0], molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        display(SVG(svg.replace("svg:", "")))

    def visualize_3d(self, mols_gen: List[Chem.Mol]) -> None:
        """Visualize 3D output.

        Args:
            mols_gen: list of generated conformations.
        """
        show(mols_gen[0])
