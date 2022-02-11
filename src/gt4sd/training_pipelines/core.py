"""Core training utilities."""

from dataclasses import dataclass


class TrainingPipeline:
    """Abstract interface for a training pipelines."""

    def train(self, **kwargs) -> None:
        """Train the models associated to a pipeline."""
        raise NotImplementedError("Can't train an abstract training pipeline.")


@dataclass
class TrainingPipelineArguments:
    """Abstract interface for training pipeline arguments."""

    __name__ = "training_args"
