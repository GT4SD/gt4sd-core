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
"""Implementation of the KeyBERT keyword extractor."""

import os
from typing import List, Optional, Union

import torch
from keybert import KeyBERT as KeyBERTCore
from sentence_transformers import SentenceTransformer

from ....frameworks.torch import device_claim


class KeyBERT:
    """
    Keyword extractor based on [KeyBERT](https://github.com/MaartenGr/KeyBERT).
    """

    def __init__(
        self,
        resources_path: str,
        minimum_keyphrase_ngram: int,
        maximum_keyphrase_ngram: int,
        stop_words: Optional[str],
        top_n: int,
        use_maxsum: bool,
        use_mmr: bool,
        diversity: float,
        number_of_candidates: int,
        model_name: str,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Initialize KeyBERT.

        Args:
            resources_path: path where to load hypothesis, candidate labels and, optionally, the model.
            minimum_keyphrase_ngram: lower bound for phrase size.
            maximum_keyphrase_ngram: upper bound for phrase size.
            stop_words: language for the stop words removal. If not provided, no stop words removal.
            top_n: number of keywords to extract.
            use_maxsum: control usage of max sum similarity for keywords generated.
            use_mmr: control usage of max marginal relevance for keywords generated.
            diversity: diversity for the results when enabling use_mmr.
            number_of_candidates: candidates considered when enabling use_maxsum.
            model_name: name of the model to load from the cache or download from SentenceTransformers.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)
        self.resources_path = resources_path
        self.minimum_keyphrase_ngram = minimum_keyphrase_ngram
        self.maximum_keyphrase_ngram = maximum_keyphrase_ngram
        self.stop_words = stop_words
        self.top_n = top_n
        self.use_maxsum = use_maxsum
        self.use_mmr = use_mmr
        self.diversity = diversity
        self.number_of_candidates = number_of_candidates
        self.model_name = model_name
        self.load_model()

    def load_model(self) -> None:
        """Load KeyBERT model."""
        if (
            os.path.exists(self.resources_path)
            and len(os.listdir(self.resources_path)) > 0
        ):
            model_name_or_path = self.resources_path
        else:
            model_name_or_path = self.model_name
        sentence_model = SentenceTransformer(model_name_or_path, device=self.device)
        self.model = KeyBERTCore(model=sentence_model)

    def predict(self, text: str) -> List[str]:
        """Get keywords sorted by relevance.

        Args:
            text: text to extract keywords from.

        Returns:
            keywords sorted by score from highest to lowest.
        """
        return [
            keyword
            for keyword, _ in self.model.extract_keywords(
                text,
                keyphrase_ngram_range=(
                    self.minimum_keyphrase_ngram,
                    self.maximum_keyphrase_ngram,
                ),
                stop_words=self.stop_words,
                top_n=self.top_n,
                use_maxsum=self.use_maxsum,
                use_mmr=self.use_mmr,
                diversity=self.diversity,
                nr_candidates=self.number_of_candidates,
            )
        ]
