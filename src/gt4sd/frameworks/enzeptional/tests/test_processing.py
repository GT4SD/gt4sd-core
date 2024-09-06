#
# MIT License
#
# Copyright (c) 2024 GT4SD team
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
"""Enzeptional processing tests."""
import pytest
import numpy as np
from gt4sd.frameworks.enzeptional import (
    HuggingFaceModelLoader,
    HuggingFaceTokenizerLoader,
    HuggingFaceEmbedder,
    sanitize_intervals,
    round_up,
    sanitize_intervals_with_padding,
    SelectionGenerator,
    CrossoverGenerator,
)


@pytest.fixture
def huggingface_embedder():
    model_loader = HuggingFaceModelLoader()
    tokenizer_loader = HuggingFaceTokenizerLoader()

    language_model_path = "facebook/esm2_t33_650M_UR50D"
    tokenizer_path = "facebook/esm2_t33_650M_UR50D"
    cache_dir = None
    device = "cpu"

    embedder = HuggingFaceEmbedder(
        model_loader,
        tokenizer_loader,
        language_model_path,
        tokenizer_path,
        cache_dir,
        device,
    )
    return embedder


def test_huggingface_embedder(huggingface_embedder):
    protein_sequences = ["MTEITAAMVKELRESTGAGMMDCKNALSETQHEEIAFLKRLME"]
    embeddings = huggingface_embedder.embed(protein_sequences)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(protein_sequences)


def test_sanitize_intervals():
    intervals = [(1, 5), (3, 7), (8, 10)]
    sanitized = sanitize_intervals(intervals)
    assert sanitized == [(1, 7), (8, 10)]


def test_round_up():
    number = 3.14
    rounded = round_up(number)
    assert rounded == 4


def test_sanitize_intervals_with_padding():
    intervals = [(1, 3), (6, 8)]
    padded_intervals = sanitize_intervals_with_padding(
        intervals, pad_value=5, max_value=20
    )
    assert padded_intervals == [(0, 4), (5, 9)]


def test_selection_generator():
    pool_of_sequences = [
        {"sequence": "A", "score": 0.9},
        {"sequence": "B", "score": 0.8},
        {"sequence": "C", "score": 0.95},
        {"sequence": "D", "score": 0.7},
    ]
    generator = SelectionGenerator()
    selected = generator.selection(pool_of_sequences, k=0.5)
    assert len(selected) == 2
    assert selected[0]["sequence"] == "C"


def test_crossover_generator():
    generator = CrossoverGenerator(threshold_probability=0.5)
    seq_a = "AAAAAAAA"
    seq_b = "BBBBBBBB"
    offspring_a, offspring_b = generator.uniform_crossover(seq_a, seq_b)
    assert len(offspring_a) == len(seq_a)
    assert all(c in ["A", "B"] for c in offspring_a)
