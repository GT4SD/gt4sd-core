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
"""Data processing utilities."""

import inspect
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from torch import nn


class PrimarySequenceEncoder(nn.Module):
    """Model like class to create tape embeddings/encodings.

    This follows tapes implementation via `run_embed` closely, but removes
    any seed/device/cuda handling (of model and batch). This can be done in
    the training loop like for any other nn.Module.

    Example:
        An example use with protein sequence dataset from `pytoda` (requires
        mock/rdkit and pytoda>0.2) passing ids with the primary sequence::

            import sys
            from mock import Mock
            sys.modules['rdkit'] = Mock()
            sys.modules['rdkit.Chem'] = Mock()
            from torch.utils.data import DataLoader
            from pytoda.datasets.protein_sequence_dataset import protein_sequence_dataset
            from pytoda.datasets.tests.test_protein_sequence_dataset import (
                FASTA_CONTENT_GENERIC, TestFileContent
            )
            from pytoda.datasets.utils import keyed

            with TestFileContent(FASTA_CONTENT_GENERIC) as a_test_file:
                sequence_dataset = keyed(protein_sequence_dataset(
                    a_test_file.filename, filetype='.fasta', backend='lazy'
                ))
                batch_size = 5
                dataloader = DataLoader(sequence_dataset, batch_size=batch_size)

                encoder = PrimarySequenceEncoder(
                    model_type='transformer',
                    from_pretrained='bert-base',
                    tokenizer='iupac',
                    log_level=logging.INFO,
                )
                # sending encoder to cuda device should work, not tested

                loaded = next(iter(dataloader))
                print(loaded)
                encoded, ids = encoder.forward(loaded)
                print(ids)
                print(encoded)

        However the forward call supports also not passing ids, but batch still
        has to be wrapped as list (of length 1)::

            encoded, dummy_ids = PrimarySequenceEncoder().forward(
                [
                    ['MQNP', 'LLLLL'],  # type: Sequence[str]
                    # sequence_ids may be missing here
                ]
            )
    """

    def __init__(
        self,
        model_type: str = "transformer",
        from_pretrained: Optional[str] = "bert-base",
        model_config_file: Optional[str] = None,
        # full_sequence_embed: bool = False,
        tokenizer: str = "iupac",
    ):
        """Initialize the PrimarySequenceEncoder.

        Args:
            model_type: Which type of model to create
                (e.g. transformer, unirep, ...). Defaults to 'transformer'.
            from_pretrained: either
                a string with the `shortcut name` of a pre-trained model to
                load from cache or download, e.g.: ``bert-base-uncased``, or
                a path to a `directory` containing model weights saved using
                :func:`tape.models.modeling_utils.ProteinConfig.save_pretrained`,
                e.g.: ``./my_model_directory/``.
                Defaults to 'bert-base'.
            model_config_file: A json config file
                that specifies hyperparameters. Defaults to None.
            tokenizer: vocabulary name. Defaults to 'iupac'.

        Note:
            tapes default seed would be 42 (see `tape.utils.set_random_seeds`)
        """
        super().__init__()
        # padding during forward goes through cpu (numpy)
        self.device_indicator = nn.Parameter(torch.empty(0), requires_grad=False)
        # dummy sequence_ids, so they are optional
        self.next_dummy_id = 0

        task_spec = registry.get_task_spec("embed")  # task = 'embed'
        # from tape.datasets import EmbedDataset
        self.model = registry.get_task_model(
            model_type, task_spec.name, model_config_file, from_pretrained
        )

        # to filter out batch items that aren't used in this model
        # see `from_collated_batch` and `tape.training.ForwardRunner`
        forward_arg_keys = inspect.getfullargspec(self.model.forward).args
        self._forward_arg_keys = forward_arg_keys[1:]  # remove self argument
        assert "input_ids" in self._forward_arg_keys

        self.tokenizer = TAPETokenizer(vocab=tokenizer)
        self.full_sequence_embed = False

        self.eval()

    def train(self, mode: bool):  # type:ignore
        """Avoid any setting to train mode."""
        return super().train(False)

    def generate_tokenized(
        self, batch: List[Sequence[str]]
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        # batch is list of len 2 (typically tuples[str] of length `batch_size`)
        for item, sequence_id in zip(*batch):
            token_ids = self.tokenizer.encode(item)
            input_mask: np.ndarray = np.ones_like(token_ids)
            yield sequence_id, token_ids, input_mask

    @classmethod
    def collate_fn(
        cls, batch: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        # from tape.datasets.EmbedDataset because there it's not a classmethod
        ids, tokens, input_mask = zip(*batch)
        ids_list: List[str] = list(ids)
        tokens_tensor: torch.Tensor = torch.from_numpy(pad_sequences(tokens))
        input_mask_tensor: torch.Tensor = torch.from_numpy(pad_sequences(input_mask))
        # on cpu now, is unavoidable as tokenizer and mask are in numpy.
        return {
            "ids": ids_list,
            "input_ids": tokens_tensor,
            "input_mask": input_mask_tensor,
        }  # type: ignore

    def from_collated_batch(
        self, batch: Dict[str, Union[List[str], torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # filter arguments
        batch_tensors: Dict[str, torch.Tensor] = {
            name: tensor  # type:ignore
            for name, tensor in batch.items()
            if name in self._forward_arg_keys
        }
        device = self.device_indicator.device
        if device.type == "cuda":
            batch_tensors = {
                name: tensor.cuda(device=device, non_blocking=True)
                for name, tensor in batch_tensors.items()
            }
        return batch_tensors

    def forward(  # type:ignore
        self, batch: List[Sequence[str]]
    ) -> Tuple[torch.Tensor, List[str]]:
        # batch: List[(primary_sequences,), (sequence_ids,))] of length 2
        # keys can be passed on by pytoda via keyed(ds: Keydataset[str])
        if len(batch) == 1:
            # no sequence_ids passed
            dummy_ids = self.get_dummy_ids(length=len(batch[0]))
            batch.append(dummy_ids)
        elif len(batch) == 2:
            pass
        else:
            raise ValueError(
                "batch should be of length 1 or 2, containing `primary_sequences` "
                " and optionally `sequence_ids`."
            )

        with torch.no_grad():
            # Iterator[(sequence_id, token_ids, input_mask)]
            batch_loader_like = self.generate_tokenized(batch)
            batch_dict_with_ids: Dict[
                str, Union[List[str], torch.Tensor]
            ] = self.collate_fn(list(batch_loader_like))
            ids: List[str] = cast(List[str], batch_dict_with_ids["ids"])
            batch_dict = self.from_collated_batch(batch_dict_with_ids)
            # outputs = self.model(**batch_dict)
            # pooled_embed = outputs[1]
            sequence_embed = self.model(**batch_dict)[0]
            sequence_lengths = batch_dict["input_mask"].sum(1)

            # can variable length slicing be done on the batch?
            if not self.full_sequence_embed:
                sequences_out: torch.Tensor = sequence_embed.new_empty(
                    # dimension of sequence length will be averaged out
                    size=sequence_embed.shape[::2]
                )
            else:
                raise NotImplementedError

            for i, (seqembed, length) in enumerate(
                zip(
                    sequence_embed,
                    sequence_lengths,
                )
            ):
                seqembed = seqembed[: int(length)]
                if not self.full_sequence_embed:
                    seqembed = seqembed.mean(0)
                sequences_out[i, ...] = seqembed

        return sequences_out, ids

    def get_dummy_ids(self, length: int) -> Tuple[str, ...]:
        first = self.next_dummy_id
        self.next_dummy_id += length  # before last
        return tuple(map(str, range(first, self.next_dummy_id)))
