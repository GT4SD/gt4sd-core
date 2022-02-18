"""CLaSS implementation."""
import logging
from typing import List, Optional

import torch

from ....extras import EXTRAS_ENABLED

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if EXTRAS_ENABLED:
    from cog.sample_pipeline import CogMolFiles, get_new_samples, load_vae, mogQ
    from cog.z_classifier import (
        load_z_ba_regressor_with_tape_embeddings,
        load_zregressor_model,
    )
    from pag.sample_pipeline import PAGFiles, filter_heuristic
    from pag.z_classifier import load_lumo_clf_model

    class UnsupportedTargetError(RuntimeError):
        """Error for target sequence with unknown embedding."""

        def __init__(self, title: str, detail: str) -> None:
            """Initialize UnsupportedTargetError.

            Args:
                title: title of the error.
                detail: description of the error.
            """
            self.type = "UnsupportedTargetError"
            self.title = title
            self.detail = detail
            super().__init__(detail)

    class CogMolGenerator:
        def __init__(
            self,
            protein_sequence: str,
            model_files: CogMolFiles,
            n_samples_per_round: int = 32,
            device: str = "cpu",
            # dropout: float = 0.2,
            # dropout: Dropout is disabled in eval mode. Defaults to 0.2.
            num_proteins_selectivity: int = 10,
            temp: float = 1.0,
            max_len: int = 100,
            bindingdb_date: Optional[str] = None,
        ) -> None:
            """CogMol generator.

            Args:
                protein_sequence: the target sequence for which to generate molecules.
                n_samples_per_round: batch size.
                model_files: dedicated NamedTuple for artifact filepaths.
                device: for example 'cpu'.
                num_proteins_selectivity: number of random samples for measuring
                    selectivity. Defaults to 10.

            Raises:
                RuntimeError: in the case extras are disabled.
            """
            if not EXTRAS_ENABLED:
                raise RuntimeError(
                    "Can't instantiate CogMolGenerator, extras disabled!"
                )

            self.n_samples_per_round = n_samples_per_round
            self.temp = temp
            self.max_len = max_len
            self.num_proteins_selectivity = num_proteins_selectivity
            self.bindingdb_date = bindingdb_date
            self.device = device

            self.model = load_vae(
                model_files.vae_model,
                model_files.vae_config,
                model_files.vae_vocab,
                device,
            )

            self.clf = load_z_ba_regressor_with_tape_embeddings(
                model_path=model_files.ba_model_path,
                device=device,
                dims=[2048, 1],
                dropout=0.2,
            )
            self.reg = load_zregressor_model(
                model_path=model_files.qed_regressor_model_path, device=device
            )

            self.protein_z_map = torch.load(f=model_files.protein_z_map)  # device?
            self.protein_emb = self.get_target_embedding(
                protein_sequence=protein_sequence
            )
            self.protein_sequence = protein_sequence

            # set all models to eval
            self.model.eval()
            self.clf.eval()
            self.reg.eval()

            self.Q_xi_a = mogQ(model_files.mog_model_file, device=device)

            self.Q_xi_a.init_attr_classifiers(
                attr_clfs={
                    "binding": self.clf,
                    "qed": self.reg,
                    "non_binding": self.clf,
                },
                clf_targets={
                    "binding": 1,
                    "qed": 0,
                    "non_binding": 0,
                },
                protein_emb_binding=self.protein_emb,
                protein_embedding_map=self.protein_z_map,
                num_proteins_selectivity=self.num_proteins_selectivity,
            )

        def sample_accepted(self, target: Optional[str] = None) -> List[str]:
            if target is not None and target != self.protein_sequence:
                self.protein_sequence = target
                self.protein_emb = self.get_target_embedding(protein_sequence=target)
                self.Q_xi_a.init_attr_classifiers(
                    attr_clfs={
                        "binding": self.clf,
                        "qed": self.reg,
                        "non_binding": self.clf,
                    },
                    clf_targets={
                        "binding": 1,
                        "qed": 0,
                        "non_binding": 0,
                    },
                    protein_emb_binding=self.protein_emb,
                    protein_embedding_map=self.protein_z_map,
                    num_proteins_selectivity=self.num_proteins_selectivity,
                )

            samples = get_new_samples(
                model=self.model,
                Q=self.Q_xi_a,
                n_samples=self.n_samples_per_round,
                max_len=self.max_len,
                temp=self.temp,
            )
            return samples[samples["accept_z"] == 1]["smiles"].tolist()

        def get_target_embedding(self, protein_sequence: str) -> torch.Tensor:
            """Retrieve embedding of target or raise a dedicated exception.

            Args:
                protein_sequence: target amino acid sequence.

            Raises:
                UnsupportedTargetError: in case the embedding is not available in
                    `self.protein_z_map`.

            Returns:
                The protein embedding.
            """
            try:
                return self.protein_z_map[protein_sequence]
            except KeyError:
                detail = (
                    "The provided target is not available in this version: \n"
                    f"{protein_sequence}"
                )
                # prepend a hint on the supported target proteins
                if self.bindingdb_date is not None:
                    detail = (
                        f"Only protein sequences published in BindingDB until {self.bindingdb_date} are supported. "
                    ) + detail

                logger.warning(detail)
                raise UnsupportedTargetError(
                    title="The target protein sequence is not supported.", detail=detail
                )

    class PAGGenerator:
        def __init__(
            self,
            model_files: PAGFiles,
            n_samples_per_round: int = 32,
            device: str = "cpu",
            temp: float = 1.0,
            max_len: int = 100,
        ) -> None:
            """PAG generator.

            Args:
                n_samples_per_round: batch size.
                model_files: dedicated NamedTuple for artifact filepaths.
                device: for example 'cpu'.

            Raises:
                RuntimeError: in the case extras are disabled.
            """
            if not EXTRAS_ENABLED:
                raise RuntimeError("Can't instantiate PAGGenerator, extras disabled!")

            self.n_samples_per_round = n_samples_per_round
            self.temp = temp
            self.max_len = max_len
            self.device = device

            self.model = load_vae(
                model_files.vae_model,
                model_files.vae_config,
                model_files.vae_vocab,
                device,
            )

            self.clf = load_lumo_clf_model(model_path=model_files.lumo_clf_model_path)

            # set all models to eval
            self.model.eval()
            self.clf.eval()

            self.Q_xi_a = mogQ(model_files.mog_model_file, device=device)

            self.Q_xi_a.init_attr_classifiers(
                attr_clfs={
                    "LUMO": self.clf,
                },
                clf_targets={
                    "LUMO": 1,
                },
            )

        def sample_accepted(self) -> List[str]:
            samples = get_new_samples(
                model=self.model,
                Q=self.Q_xi_a,
                n_samples=self.n_samples_per_round,
                max_len=self.max_len,
                temp=self.temp,
            )
            # TODO: allow user input to heuristics?
            samples = filter_heuristic(samples)
            return samples[samples.accept_z & samples.accept].smiles.tolist()


else:
    logger.warning("install cogmol-inference extras to use CLaSS")
