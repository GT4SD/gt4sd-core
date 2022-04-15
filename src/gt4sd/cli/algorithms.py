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
"""Utilities for algorithms retrieval."""

from typing import Dict, List

from ..algorithms.registry import ApplicationsRegistry, ConfigurationTuple
from ..configuration import reset_logging_root_logger

reset_logging_root_logger()

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
