"""Guacamol Baselines training utilities."""

from typing import Any, Dict

from ..core import TrainingPipeline


class GuacamolBaselinesTrainingPipeline(TrainingPipeline):
    """Guacamol Baselines training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for Guacamol Baselines training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """
        raise NotImplementedError
