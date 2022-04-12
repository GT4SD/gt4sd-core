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
