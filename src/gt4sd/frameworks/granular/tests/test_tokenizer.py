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
"""Tests for granular tokenizer."""

from gt4sd.frameworks.granular.tokenizer.tokenizer import (
    BigSmilesTokenizer,
    SelfiesTokenizer,
    SmilesTokenizer,
)


def test_tokenization():
    smiles = [
        "c1ccccc1",
        "c1ccc(CP(c2ccccc2)c2ccccc2)cc1.CCCCN1[C]N(Cc2ccccc2)c2ccccc21.[Ag]",
    ]

    def _test_tokenizer(tokenizer_type, tokens_groundtruth):
        tokenizer = tokenizer_type("test", smiles=smiles)
        tokens = tokenizer.tokenize(smiles[1])
        assert tokens_groundtruth == tokens
        assert [
            tokenizer.vocab[token] for token in tokens
        ] == tokenizer.convert_tokens_to_ids(tokenizer.tokenize(smiles[1]))
        assert 2 == len(
            [
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(a_smiles))
                for a_smiles in smiles
            ]
        )

    _test_tokenizer(
        SelfiesTokenizer,
        [
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[Branch1_3]",
            "[=S]",
            "[C]",
            "[P]",
            "[Branch1_3]",
            "[Branch2_2]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[Ring1]",
            "[Branch1_1]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[Ring1]",
            "[Branch1_1]",
            "[c]",
            "[c]",
            "[Ring1]",
            "[#C]",
            "[.]",
            "[C]",
            "[C]",
            "[C]",
            "[C]",
            "[N]",
            "[Cexpl]",
            "[N]",
            "[Branch1_3]",
            "[Branch2_3]",
            "[C]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[Ring1]",
            "[Branch1_1]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[c]",
            "[Ring1]",
            "[Branch1_1]",
            "[Ring1]",
            "[=N]",
            "[.]",
            "[Agexpl]",
        ],
    )

    _test_tokenizer(
        SmilesTokenizer,
        [
            "c",
            "1",
            "c",
            "c",
            "c",
            "(",
            "C",
            "P",
            "(",
            "c",
            "2",
            "c",
            "c",
            "c",
            "c",
            "c",
            "2",
            ")",
            "c",
            "2",
            "c",
            "c",
            "c",
            "c",
            "c",
            "2",
            ")",
            "c",
            "c",
            "1",
            ".",
            "C",
            "C",
            "C",
            "C",
            "N",
            "1",
            "[C]",
            "N",
            "(",
            "C",
            "c",
            "2",
            "c",
            "c",
            "c",
            "c",
            "c",
            "2",
            ")",
            "c",
            "2",
            "c",
            "c",
            "c",
            "c",
            "c",
            "2",
            "1",
            ".",
            "[Ag]",
        ],
    )


def test_big_smiles_tokenization():
    big_smiles = "{[][$]CC(C#N)[$],[$]CC(c1ccccc1)[$][]}"
    tokenizer = BigSmilesTokenizer("test")
    assert tokenizer.tokenize(big_smiles) == [
        "{",
        "[]",
        "[$]",
        "C",
        "C",
        "(",
        "C",
        "#",
        "N",
        ")",
        "[$]",
        ",",
        "[$]",
        "C",
        "C",
        "(",
        "c",
        "1",
        "c",
        "c",
        "c",
        "c",
        "c",
        "1",
        ")",
        "[$]",
        "[]",
        "}",
    ]
