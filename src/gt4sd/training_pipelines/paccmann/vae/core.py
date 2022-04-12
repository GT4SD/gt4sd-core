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
"""PaccMann VAE training utilities."""

import json
import logging
import os
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional, cast

import torch
from paccmann_chemistry.models.training import train_vae
from paccmann_chemistry.models.vae import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils import collate_fn, disable_rdkit_logging
from paccmann_chemistry.utils.hyperparams import SEARCH_FACTORY
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from torch.utils.tensorboard import SummaryWriter

from ....frameworks.torch import get_device
from ...core import TrainingPipelineArguments
from ..core import PaccMannTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PaccMannVAETrainingPipeline(PaccMannTrainingPipeline):
    """Language modeling training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for PaccMann training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        """
        try:
            device = get_device()
            disable_rdkit_logging()
            params = {**training_args, **dataset_args, **model_args}
            train_smiles_filepath = params["train_smiles_filepath"]
            test_smiles_filepath = params["test_smiles_filepath"]
            smiles_language_filepath = params.get("smiles_language_filepath", None)
            params["batch_mode"] = "Padded"

            model_path = params["model_path"]
            training_name = params["training_name"]

            writer = SummaryWriter(f"logs/{training_name}")
            logger.info(f"Model with name {training_name} starts.")

            model_dir = os.path.join(model_path, training_name)
            log_path = os.path.join(model_dir, "logs")
            val_dir = os.path.join(log_path, "val_logs")
            os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
            os.makedirs(log_path, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Load SMILES language
            smiles_language: Optional[SMILESLanguage] = None
            if smiles_language_filepath is not None:
                smiles_language = SMILESLanguage.load(smiles_language_filepath)

            logger.info(f"Smiles filepath: {train_smiles_filepath}")

            # create SMILES eager dataset
            smiles_train_data = SMILESDataset(
                train_smiles_filepath,
                smiles_language=smiles_language,
                padding=False,
                selfies=params.get("selfies", False),
                add_start_and_stop=params.get("add_start_stop_token", True),
                augment=params.get("augment_smiles", False),
                canonical=params.get("canonical", False),
                kekulize=params.get("kekulize", False),
                all_bonds_explicit=params.get("all_bonds_explicit", False),
                all_hs_explicit=params.get("all_hs_explicit", False),
                remove_bonddir=params.get("remove_bonddir", False),
                remove_chirality=params.get("remove_chirality", False),
                backend="lazy",
                device=device,
            )
            smiles_test_data = SMILESDataset(
                test_smiles_filepath,
                smiles_language=smiles_language,
                padding=False,
                selfies=params.get("selfies", False),
                add_start_and_stop=params.get("add_start_stop_token", True),
                augment=params.get("augment_smiles", False),
                canonical=params.get("canonical", False),
                kekulize=params.get("kekulize", False),
                all_bonds_explicit=params.get("all_bonds_explicit", False),
                all_hs_explicit=params.get("all_hs_explicit", False),
                remove_bonddir=params.get("remove_bonddir", False),
                remove_chirality=params.get("remove_chirality", False),
                backend="lazy",
                device=device,
            )

            if smiles_language_filepath is None:
                smiles_language = smiles_train_data.smiles_language
                smiles_language.save(os.path.join(model_path, f"{training_name}.lang"))
            else:
                smiles_language_filename = os.path.basename(smiles_language_filepath)
                cast(SMILESLanguage, smiles_language).save(
                    os.path.join(model_dir, smiles_language_filename)
                )

            params.update(
                {
                    "vocab_size": cast(
                        SMILESLanguage, smiles_language
                    ).number_of_tokens,
                    "pad_index": cast(SMILESLanguage, smiles_language).padding_index,
                }
            )

            vocab_dict = cast(SMILESLanguage, smiles_language).index_to_token
            params.update(
                {
                    "start_index": list(vocab_dict.keys())[
                        list(vocab_dict.values()).index("<START>")
                    ],
                    "end_index": list(vocab_dict.keys())[
                        list(vocab_dict.values()).index("<STOP>")
                    ],
                }
            )

            if params.get("embedding", "learned") == "one_hot":
                params.update({"embedding_size": params["vocab_size"]})

            with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
                json.dump(params, fp)

            # create DataLoaders
            train_data_loader = torch.utils.data.DataLoader(
                smiles_train_data,
                batch_size=params.get("batch_size", 64),
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=True,
                pin_memory=params.get("pin_memory", True),
                num_workers=params.get("num_workers", 8),
            )

            test_data_loader = torch.utils.data.DataLoader(
                smiles_test_data,
                batch_size=params.get("batch_size", 64),
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=True,
                pin_memory=params.get("pin_memory", True),
                num_workers=params.get("num_workers", 8),
            )
            # initialize encoder and decoder
            gru_encoder = StackGRUEncoder(params).to(device)
            gru_decoder = StackGRUDecoder(params).to(device)
            gru_vae = TeacherVAE(gru_encoder, gru_decoder).to(device)
            logger.info("Model summary:")
            for name, parameter in gru_vae.named_parameters():
                logger.info(f"Param {name}, shape:\t{parameter.shape}")
            total_params = sum(p.numel() for p in gru_vae.parameters())
            logger.info(f"Total # params: {total_params}")

            loss_tracker = {
                "test_loss_a": 10e4,
                "test_rec_a": 10e4,
                "test_kld_a": 10e4,
                "ep_loss": 0,
                "ep_rec": 0,
                "ep_kld": 0,
            }

            # train for n_epoch epochs
            logger.info("Model creation and data processing done, Training starts.")
            decoder_search = SEARCH_FACTORY[params.get("decoder_search", "sampling")](
                temperature=params.get("temperature", 1.0),
                beam_width=params.get("beam_width", 3),
                top_tokens=params.get("top_tokens", 5),
            )

            if writer:
                pparams = params.copy()
                pparams["training_file"] = train_smiles_filepath
                pparams["test_file"] = test_smiles_filepath
                pparams["language_file"] = smiles_language_filepath
                pparams["model_path"] = model_path
                pparams = {k: v if v is not None else "N.A." for k, v in params.items()}
                pparams["training_name"] = training_name
                from pprint import pprint

                pprint(pparams)
                writer.add_hparams(hparam_dict=pparams, metric_dict={})

            for epoch in range(params["epochs"] + 1):
                t = time()
                loss_tracker = train_vae(
                    epoch,
                    gru_vae,
                    train_data_loader,
                    test_data_loader,
                    smiles_language,
                    model_dir,
                    search=decoder_search,
                    optimizer=params.get("optimizer", "adadelta"),
                    lr=params["learning_rate"],
                    kl_growth=params["kl_growth"],
                    input_keep=params["input_keep"],
                    test_input_keep=params["test_input_keep"],
                    generate_len=params["generate_len"],
                    log_interval=params["log_interval"],
                    save_interval=params["save_interval"],
                    eval_interval=params["eval_interval"],
                    loss_tracker=loss_tracker,
                    logger=logger,
                    batch_mode=params["batch_mode"],
                )
                logger.info(f"Epoch {epoch}, took {time() - t:.1f}.")

            logger.info(
                "Overall:\tBest loss = {0:.4f} in Ep {1}, "
                "best Rec = {2:.4f} in Ep {3}, "
                "best KLD = {4:.4f} in Ep {5}".format(
                    loss_tracker["test_loss_a"],
                    loss_tracker["ep_loss"],
                    loss_tracker["test_rec_a"],
                    loss_tracker["ep_rec"],
                    loss_tracker["test_kld_a"],
                    loss_tracker["ep_kld"],
                )
            )
            logger.info("Training done, shutting down.")
        except Exception:
            logger.exception(
                "Exception occurred while running PaccMannVAETrainingPipeline"
            )


@dataclass
class PaccMannVAEModelArguments(TrainingPipelineArguments):
    """Arguments pertaining to model instantiation."""

    __name__ = "model_args"

    n_layers: int = field(
        default=2, metadata={"help": "Number of layers for the RNNs."}
    )
    bidirectional: bool = field(
        default=False, metadata={"help": "Whether the RNN cells are bidirectional."}
    )
    rnn_cell_size: int = field(default=512, metadata={"help": "Size of the RNN cells."})
    latent_dim: int = field(default=256, metadata={"help": "Size of the RNN cells."})
    stack_width: int = field(
        default=50, metadata={"help": "Width of the memory stack for the RNN cell."}
    )
    stack_depth: int = field(
        default=50, metadata={"help": "Depth of the memory stack for the RNN cell."}
    )
    decode_search: str = field(
        default="sampling", metadata={"help": "Decoder search strategy."}
    )
    dropout: float = field(default=0.2, metadata={"help": "Dropout rate to apply."})
    generate_len: int = field(
        default=100, metadata={"help": "Length in tokens of the generated molecules."}
    )
    kl_growth: float = field(
        default=0.003, metadata={"help": "Growth of the KL term weight in the loss."}
    )
    input_keep: float = field(
        default=0.85, metadata={"help": "Probability to keep input tokens in train."}
    )
    test_input_keep: float = field(
        default=1.0, metadata={"help": "Probability to keep input tokens in test."}
    )
    temperature: float = field(
        default=0.8, metadata={"help": "Temperature for the sampling."}
    )
    embedding: str = field(
        default="one_hot",
        metadata={
            "help": "Embedding technique for the tokens. 'one_hot' or 'learned'."
        },
    )
    vocab_size: int = field(
        default=380, metadata={"help": "Size of the vocabulary of chemical tokens."}
    )
    pad_index: int = field(default=0, metadata={"help": "Index for the padding token."})
    embedding_size: int = field(
        default=380, metadata={"help": "Size of the embedding vectors."}
    )
    beam_width: int = field(default=3, metadata={"help": "Width of the beam search."})
    top_tokens: int = field(
        default=5, metadata={"help": "Number of tokens to consider in the beam search."}
    )
