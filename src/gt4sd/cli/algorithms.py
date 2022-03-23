"""Utilities for algorithms retrieval."""

from typing import Dict, List

from ..algorithms.registry import ApplicationsRegistry, ConfigurationTuple

AVAILABLE_ALGORITHMS = sorted(
    ApplicationsRegistry.list_available(),
    key=lambda algorithm: (
        algorithm["domain"],
        algorithm["algorithm_type"],
        algorithm["algorithm_name"],
        algorithm["algorithm_application"],
        algorithm["algorithm_version"],
    ),
)
AVAILABLE_ALGORITHMS_CATEGORIES = {
    category: sorted(set([algorithm[category] for algorithm in AVAILABLE_ALGORITHMS]))
    for category in ["domain", "algorithm_type"]
}


def filter_algorithm_applications(
    algorithms: List[Dict[str, str]], filters: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Returning algorithms with given filters.

    Args:
        algorithms (List[Dict[str, str]]): a list of algorithm applications as dictionaries.
        filters (Dict[str, str]): the filters to apply.

    Returns:
        the filtered algorithms.
    """
    return [
        application
        for application in algorithms
        if all(
            [
                application[filter_type] == filter_value if filter_value else True
                for filter_type, filter_value in filters.items()
            ]
        )
    ]


def get_configuration_tuples(
    algorithms: List[Dict[str, str]]
) -> List[ConfigurationTuple]:
    """
    Returning configuration tuples from a list of applications.

    Args:
        algorithms (List[Dict[str, str]]): a list of algorithm applications as dictionaries.

    Returns:
        the configuration tuples.
    """
    return sorted(
        list(
            {
                ConfigurationTuple(
                    **{
                        key: value
                        for key, value in algorithm.items()
                        if key in ConfigurationTuple.__annotations__.keys()
                    }
                )
                for algorithm in algorithms
            }
        )
    )
