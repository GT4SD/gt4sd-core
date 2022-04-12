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
"""Parsing utilties."""

import argparse
import ast
from typing import Union


def str2bool(s: Union[str, bool]) -> bool:
    """Convert a string into a bool.

    Args:
        s: a string representation of a boolean.

    Raises:
        argparse.ArgumentTypeError: in case the conversion is failing.

    Returns:
        the converted value.
    """
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def convert_string_to_class(s: str):
    """Convert a string into a python object.

    Fallback to ast in case of unexpected strings.

    Args:
        s: a string.

    Returns:
        the converted python object.
    """
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    elif s.lower() == "none":
        return None
    elif s:
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
