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
"""Functions to facilitate conversion from dataclasses to training descriptions."""

import logging
from dataclasses import _MISSING_TYPE, fields
from typing import Any, Dict, Optional, Type, Union

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def find_type(input_type: Type) -> Optional[str]:
    """Convert type class to string.

    Args:
        input_type: type to be converted to string.

    Returns:
        string of the type or None in case the given type is not supported.
    """
    field_type = None
    if input_type is str:
        field_type = "string"
    elif input_type is int:
        field_type = "integer"
    elif input_type is float:
        field_type = "number"
    elif input_type is bool:
        field_type = "boolean"

    return field_type


def extract_fields_from_class(
    dataclass: Type,
) -> Dict[str, Any]:
    """Extract arguments from dataclass.

    Args:
        dataclass: Dataclass to contains the arguments.

    Returns:
        Dictionary of the existing arguments including their type, description and default value.
    """

    # assign type and description
    arg_fields = {
        field.name: {
            "type": field.type,
            "description": field.metadata.get("help", "No help provided"),
        }
        for field in fields(dataclass)
    }

    # assign default values
    for field in fields(dataclass):

        if not isinstance(field.default, _MISSING_TYPE):

            if field.default is None:
                field.default = "none"

            arg_fields[field.name]["default"] = field.default

    # convert type to str
    for field_name in arg_fields:

        field_type = find_type(arg_fields[field_name]["type"])

        if field_type:

            arg_fields[field_name]["type"] = field_type

        elif (
            hasattr(arg_fields[field_name]["type"], "__origin__")
            and arg_fields[field_name]["type"].__origin__ is Union
        ):

            types = [
                find_type(type) for type in arg_fields[field_name]["type"].__args__
            ]
            types = [type for type in types if type is not None]

            if len(types) == 1:
                arg_fields[field_name]["type"] = types[0]
            else:
                raise ValueError(f"{arg_fields[field_name]['type']} not supported")

        else:
            # NOTE: not raising since the HF training args might introduce typing inconsistencies
            # Could be changed once the following is merged: https://github.com/huggingface/transformers/pull/17934
            logger.error(
                f" argument {field_name}: {arg_fields[field_name]['type']} not supported"
            )

    return arg_fields
