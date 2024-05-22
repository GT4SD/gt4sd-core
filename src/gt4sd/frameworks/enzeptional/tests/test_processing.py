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
from gt4sd.frameworks.enzeptional.processing import (
    ModelCache,
    get_device,
    sanitize_intervals,
    sanitize_intervals_with_padding,
    reconstruct_sequence_with_mutation_range,
)
import torch


def test_add_and_get_model():
    model_cache = ModelCache()
    test_model = torch.nn.Module()
    model_cache.add("test_model", test_model)
    retrieved_model = model_cache.get("test_model")
    assert test_model == retrieved_model


class TestUtilityFunctions:
    def test_get_device(self):
        expected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert str(get_device()) == expected_device

    def test_sanitize_intervals(self):
        intervals = [(1, 3), (2, 5), (6, 8)]
        sanitized = sanitize_intervals(intervals)
        assert sanitized == [(1, 5), (6, 8)]

    def test_sanitize_intervals_with_padding(self):
        intervals = [(1, 3), (6, 8)]
        padded_intervals = sanitize_intervals_with_padding(intervals, 8, 50)
        assert padded_intervals == [(0, 11)]

    def test_reconstruct_sequence_with_mutation_range(self):
        original_sequence = "AACCGGTT"
        mutation_range = "NNNN"
        intervals = [(2, 4), (6, 8)]
        reconstructed = reconstruct_sequence_with_mutation_range(
            original_sequence, mutation_range, intervals
        )
        assert reconstructed == "AANNGGNN"
