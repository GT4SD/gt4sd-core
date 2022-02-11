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
