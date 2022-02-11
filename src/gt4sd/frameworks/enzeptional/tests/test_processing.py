"""Enzeptional processing tests."""

from gt4sd.frameworks.enzeptional.processing import (
    reconstruct_sequence_with_mutation_range,
    sanitize_intervals,
)


def test_sanitize_intervals():
    assert sanitize_intervals([(-5, 12), (13, 14), (2, 3), (-3, 4), (-2, 6)]) == [
        (-5, 12),
        (13, 14),
    ]


def test_reconstruct_sequence_with_mutation_range():
    assert (
        reconstruct_sequence_with_mutation_range(
            "ABCDEFGHILMNOPQRSTUVWXYZ", "12789", [(0, 1), (6, 8)]
        )
        == "12CDEF789LMNOPQRSTUVWXYZ"
    )
