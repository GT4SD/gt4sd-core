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
from typing import List


def get_substructure_indices(
    full_sequence: List[str], substructure: List[str]
) -> List[int]:
    """
    Args:
        full_sequence: A list of strings, each representing a token from the full sequence
        substructure: A list of strings, each representing a token from the substructure that
            is contained in the full sequence.

    Returns:
        A list of integers, corresponding to all the indices of the tokens in the full sequence
        that match the substructure.

    """
    substructure_indices: List = []
    for i in range(len(full_sequence)):
        if full_sequence[i] == substructure[0]:
            if full_sequence[i : i + len(substructure)] == substructure:
                substructure_indices.extend(range(i, i + len(substructure)))
    return substructure_indices
