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
"""Tokenizers implementations."""

import collections
import logging
import os
from typing import Dict, Iterable, List, Type

import regex as re
import selfies as sf
from pytoda.smiles.processing import tokenize_selfies

SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
BIG_SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\,|\{|\}|\[\]|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def selfies_alphabet() -> List[str]:
    """Legacy selfies 0.2.4 alphabet method.

    Adapted from: https://github.com/aspuru-guzik-group/selfies/blob/84122855ae76a928e1cb7d58796b8b47385a4359/selfies/selfies.py#L4.

    Returns:
        SELFIES list of tokens.
    """
    alphabet = [
        "[Branch1_1]",
        "[Branch1_2]",
        "[Branch1_3]",
        "[Ring1]",
        "[Branch2_1]",
        "[Branch2_2]",
        "[Branch2_3]",
        "[Ring2]",
        "[Branch3_1]",
        "[Branch3_2]",
        "[Branch3_3]",
        "[Ring3]",
        "[O]",
        "[=O]",
        "[N]",
        "[=N]",
        "[C]",
        "[=C]",
        "[#C]",
        "[S]",
        "[=S]",
        "[P]",
        "[F]",
        "[C@Hexpl]",
        "[C@@Hexpl]",
        "[C@expl]",
        "[C@@expl]",
        "[H]",
        "[NHexpl]",
    ]
    return alphabet


def load_vocab(vocab_file: str) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary.

    Args:
        vocab_file: vocabulary file.

    Returns:
        vocabulary mapping tokens to indices.
    """
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class BasicTokenizer:
    """Basic tokenizer."""

    def __init__(
        self,
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        """Constructs a BasicSmilesTokenizer.

        Args:
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text.

        Args:
            text: text to tokenize.

        Returns:
            list of tokens.
        """
        return list(text)

    def build_vocab(self, smiles: Iterable[str], vocab_file: str) -> List[str]:
        """Build and save a vocabulary given a SMILES list.

        Args:
            smiles: iterable of SMILES.
            vocab_file: path to a file where the vocabulary is saved.

        Returns:
            a list of all tokens in the vocabulary.
        """
        tokens = set([self.pad_token, self.sos_token, self.eos_token, self.unk_token])

        for smile in smiles:
            tokens_temp = self.tokenize(smile)

            for token in tokens_temp:
                tokens.add(token)

        tokens_list = sorted(list(tokens))

        with open(vocab_file, "w") as f:
            for item in tokens_list:
                f.write(f"{item}{os.linesep}")

        return tokens_list


class BasicSmilesTokenizer(BasicTokenizer):
    """Basic SMILES tokenizer."""

    def __init__(
        self,
        regex_pattern: str = SMI_REGEX_PATTERN,
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        """Constructs a BasicSmilesTokenizer.

        Args:
            regex_pattern: regex pattern. Defaults to SMI_REGEX_PATTERN.
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        super().__init__(
            pad_token=pad_token,
            sos_token=sos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text.

        Args:
            text: text to tokenize.

        Returns:
            list of tokens.
        """
        return [token for token in self.regex.findall(text)]


class BasicSelfiesTokenizer(BasicTokenizer):
    """Basic SELFIES tokenizer."""

    def __init__(
        self,
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        """Constructs a BasicSelfiesTokenizer.

        Args:
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

    def smiles_to_selfies(self, smiles: Iterable[str]) -> List[str]:
        """Convert a list of SMILES into SELFIES.

        Args:
            smiles: a list of SMILES.

        Returns:
            a list of SELFIES.
        """
        return [sf.encoder(a_smiles) for a_smiles in smiles]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text.

        Args:
            text: text to tokenize.

        Returns:
            list of tokens.
        """
        return tokenize_selfies(sf.encoder(text))

    def build_vocab(self, smiles: Iterable[str], vocab_file: str) -> List[str]:
        """Build and save a vocabulary given a SMILES list.

        Args:
            smiles: iterable of SMILES.
            vocab_file: path to a file where the vocabulary is saved.

        Returns:
            a list of all tokens in the vocabulary.
        """
        selfies = self.smiles_to_selfies(smiles)
        tokens = set(
            [self.pad_token, self.sos_token, self.eos_token, self.unk_token, "[.]"]
            + selfies_alphabet()
        )
        for a_selfies in selfies:
            tokens = tokens | set(tokenize_selfies(a_selfies))

        tokens_list = sorted(list(tokens))

        with open(vocab_file, "w") as f:
            for item in tokens_list:
                f.write(f"{item}{os.linesep}")

        return tokens_list


class Tokenizer:
    """Tokenizer that can build a vocabulary on the fly."""

    def __init__(
        self,
        vocab_file: str,
        basic_tokenizer: BasicTokenizer = BasicTokenizer(),
        smiles: List[str] = [],
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:

        """Constructs a Tokenizer.

        Args:
            vocab_file: path to vocabulary file. If the file is not present, the provided SMILES list
                is used to generate one.
            basic_tokenizer: a basic tokenizer. Defaults to BasicTokenizer character tokenizer.
            smiles: list of smiles. Default to empty list, used only if the vocabulary file does not exist.
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        self.basic_tokenizer = basic_tokenizer
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # load or build vocab
        if os.path.isfile(vocab_file) and len(smiles) == 0:
            logger.info(f"load vocab from: {vocab_file}")
            self.vocab = load_vocab(vocab_file)
        else:
            logger.info("build tokenizer and vocabulary")
            self.basic_tokenizer.build_vocab(smiles, vocab_file)
            logger.info(f"saved vocabulary: {vocab_file}")
        self.vocab = load_vocab(vocab_file)
        self.vocab_ids = {token: index for token, index in self.vocab.items()}
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )

        self.pad_token_id = self.vocab.get(pad_token, self.vocab[self.unk_token])
        self.sos_token_id = self.vocab.get(sos_token, self.vocab[self.unk_token])

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary.

        Returns:
            vocabulary file.
        """
        return len(self.vocab)

    @property
    def vocab_list(self) -> List[str]:
        """Return vocabulary tokens.

        Returns:
            all tokens from the vocabulary.
        """
        return list(self.vocab.keys())

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a given text.

        Args:
            text: text to tokenize.

        Returns:
            list of tokens.
        """
        return [token for token in self.basic_tokenizer.tokenize(text)]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices.

        Args:
            tokens: list of tokens.

        Returns:
            list of indices.
        """
        ids = []
        for token in tokens:
            ids.append(self.convert_token_to_id(token))
        return ids

    def convert_token_to_id(self, token: str) -> int:
        """Convert token to index.

        Args:
            token: a token.

        Returns:
            index corresponding to the input token. Unknown token index if the input
                token is not present in the vocabulary.
        """
        return self.vocab.get(token, self.vocab[self.unk_token])

    def convert_id_to_token(self, index: int) -> str:
        """Convert index to token.

        Args:
            index: an index.

        Returns:
            token corresponding to the input index. Unknown token if the input
                index is not found.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def add_padding_tokens(
        self, token_ids: List[int], length: int, right: bool = True
    ) -> List[int]:
        """Add padding token indices to the provided token indices.

        Args:
            token_ids: token indices.
            length: length of the sequence.
            right: wheter the padding is performed on the right. Defaults to True, if False
                the padding happens on the left.

        Returns:
            the padded sequence.
        """
        padding = [self.pad_token_id] * (length - len(token_ids))
        if right:
            return token_ids + padding
        else:
            return padding + token_ids


class GenericTokenizer(Tokenizer):
    """Generic tokenizer that can build a vocabulary on the fly."""

    def __init__(
        self,
        vocab_file: str,
        smiles: List[str] = [],
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        """Constructs a GenericTokenizer.

        Args:
            vocab_file: path to vocabulary file. If the file is not present, the provided SMILES list
                is used to generate one.
            smiles: list of smiles. Default to empty list, used only if the vocabulary file does not exist.
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        super().__init__(
            vocab_file=vocab_file,
            basic_tokenizer=BasicTokenizer(
                pad_token=pad_token,
                sos_token=sos_token,
                eos_token=eos_token,
                unk_token=unk_token,
            ),
            smiles=smiles,
            pad_token=pad_token,
            sos_token=sos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )


class SmilesTokenizer(Tokenizer):
    """SMILES tokenizer that can build a vocabulary on the fly."""

    def __init__(
        self,
        vocab_file: str,
        smiles: List[str] = [],
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        """Constructs a SmilesTokenizer.

        Args:
            vocab_file: path to vocabulary file. If the file is not present, the provided SMILES list
                is used to generate one.
            smiles: list of smiles. Default to empty list, used only if the vocabulary file does not exist.
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        super().__init__(
            vocab_file=vocab_file,
            basic_tokenizer=BasicSmilesTokenizer(
                pad_token=pad_token,
                sos_token=sos_token,
                eos_token=eos_token,
                unk_token=unk_token,
            ),
            smiles=smiles,
            pad_token=pad_token,
            sos_token=sos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )


class BigSmilesTokenizer(Tokenizer):
    """Big-SMILES tokenizer that can build a vocabulary on the fly."""

    def __init__(
        self,
        vocab_file: str,
        smiles: List[str] = [],
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        """Constructs a BigSmilesTokenizer.

        Args:
            vocab_file: path to vocabulary file. If the file is not present, the provided Big-SMILES list
                is used to generate one.
            smiles: list of big smiles. Default to empty list, used only if the vocabulary file does not exist.
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        super().__init__(
            vocab_file=vocab_file,
            basic_tokenizer=BasicSmilesTokenizer(
                regex_pattern=BIG_SMI_REGEX_PATTERN,
                pad_token=pad_token,
                sos_token=sos_token,
                eos_token=eos_token,
                unk_token=unk_token,
            ),
            smiles=smiles,
            pad_token=pad_token,
            sos_token=sos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )


class SelfiesTokenizer(Tokenizer):
    """SELFIES tokenizer that can build a vocabulary on the fly."""

    def __init__(
        self,
        vocab_file: str,
        smiles: List[str] = [],
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        """Constructs a SelfiesTokenizer.

        Args:
            vocab_file: path to vocabulary file. If the file is not present, the provided SMILES list
                is used to generate one.
            smiles: list of smiles. Default to empty list, used only if the vocabulary file does not exist.
            pad_token: padding token. Defaults to '<pad>'.
            sos_token: start of sequence token. Defaults to '<sos>'.
            eos_token: end of sequence token. Defaults to '</s>'.
            unk_token: unknown token. Defaults to '<unk>'.
        """
        super().__init__(
            vocab_file=vocab_file,
            basic_tokenizer=BasicSelfiesTokenizer(
                pad_token=pad_token,
                sos_token=sos_token,
                eos_token=eos_token,
                unk_token=unk_token,
            ),
            smiles=smiles,
            pad_token=pad_token,
            sos_token=sos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )


TOKENIZER_FACTORY: Dict[str, Type[Tokenizer]] = {
    "generic": GenericTokenizer,
    "smiles": SmilesTokenizer,
    "big-smiles": BigSmilesTokenizer,
    "selfies": SelfiesTokenizer,
}
