"""HuggingFaceGenerationAlgorithm initialization."""

from .core import (
    HuggingFaceCTRLGenerator,
    HuggingFaceGenerationAlgorithm,
    HuggingFaceGPT2Generator,
    HuggingFaceOpenAIGPTGenerator,
    HuggingFaceTransfoXLGenerator,
    HuggingFaceXLMGenerator,
    HuggingFaceXLNetGenerator,
)

__all__ = [
    "HuggingFaceGenerationAlgorithm",
    "HuggingFaceXLMGenerator",
    "HuggingFaceCTRLGenerator",
    "HuggingFaceGPT2Generator",
    "HuggingFaceOpenAIGPTGenerator",
    "HuggingFaceXLNetGenerator",
    "HuggingFaceTransfoXLGenerator",
]
