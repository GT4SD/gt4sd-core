"""Implementation details for a Template algorithm"""

import random
from typing import List


class Generator:
    """Basic Generator for the template algorithm"""

    def __init__(self, resources_path: str, temperature: int) -> None:
        """Initialize the Generator.

        Args:
            resources_path: directory where to find models and parameters.

        """

        self.resources_path = resources_path
        self.temperature = temperature

    def hello_name(
        self,
        name: str,
    ) -> List[str]:
        """Validate a list of strings.

        Args:
            name: a string.

        Returns:
            a list containing salutation and temperature converted to fahrenheit.
        """
        return [
            f"Hello {str(name)} {random.randint(1, int(1e6))} times and, fun fact, {str(self.temperature)} celsius equals to {(self.temperature * (9/5) + 32)} fahrenheit."
        ]
