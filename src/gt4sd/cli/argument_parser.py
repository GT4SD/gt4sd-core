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
"""Argument parser for training pipelines."""


import ast
import dataclasses
import re
from argparse import ArgumentTypeError
from enum import Enum
from functools import partial
from typing import Any, Callable, List, NewType, Optional, Type, Union

from transformers import HfArgumentParser


def eval_lambda(val: str) -> Callable:
    """Parse a lambda from a string safely.

    Args:
        val: string representing a lambda.

    Returns:
        a callable.

    Raises:
        ValueError: in case the lambda can not be parsed.
    """
    parsed_lamba = ast.parse(val).body[0].value  # type:ignore
    if isinstance(parsed_lamba, ast.Lambda) and "eval" not in val:
        return eval(val)
    else:
        raise ValueError(f"'{val}' can not be safely parsed as a lambda function")


def none_checker_bool(val: Union[bool, str]) -> Union[bool, None]:
    """Check given bool argument for None.

    Args:
        val: model arguments passed to the configuration.

    Returns:
        Bool value or None.

    Raises:
        ArgumentTypeError: value can not be parsed.
    """
    if not val:
        return None
    if isinstance(val, bool):
        return val
    if val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {val} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def none_checker(val: Any, dtype: Type) -> Any:
    """Check given argument for None.

    Args:
        val: model arguments passed to the configuration.
        dtype: expected argument type.

    Returns:
        Value casted in the expected type or None.
    """
    if not val or val == "none":
        return None
    return dtype(val)


DataClass = NewType("DataClass", Any)  # type: ignore
DataClassType = NewType("DataClassType", Any)  # type: ignore


class ArgumentParser(HfArgumentParser):
    """ArgumentParser inherited from hf's parser with modified dataclass arguments addition for better handling of None values."""

    def _add_dataclass_arguments(self, dtype: DataClassType) -> None:
        """Add a dataclass arguments.

        Args:
            dtype: data class type.
        """

        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self  # type: ignore
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()  # type: ignore
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                        typestring == f"typing.Union[{collection[prim_type]}, NoneType]"  # type: ignore
                        or typestring == f"typing.Optional[{collection[prim_type]}]"  # type: ignore
                    ):
                        field.type = collection[prim_type]  # type: ignore
                if (
                    typestring == f"typing.Union[{prim_type.__name__}, NoneType]"
                    or typestring == f"typing.Optional[{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = [x.value for x in field.type]
                kwargs["type"] = type(kwargs["choices"][0])
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            elif field.type is bool or field.type == Optional[bool]:

                if field.default is True:
                    parser.add_argument(
                        f"--no_{field.name}",
                        action="store_false",
                        dest=field.name,
                        **kwargs,
                    )

                # Hack because type=bool in argparse does not behave as we want.
                kwargs["type"] = none_checker_bool
                if field.type is bool or (
                    field.default is not None
                    and field.default is not dataclasses.MISSING
                ):
                    # Default value is False if we have no default when of type bool.
                    default = (
                        False if field.default is dataclasses.MISSING else field.default
                    )
                    # This is the value that will get picked if we don't include --field_name in any way
                    kwargs["default"] = default
                    # This tells argparse we accept 0 or 1 value after --field_name
                    kwargs["nargs"] = "?"
                    # This is the value that will get picked if we do --field_name (without value)
                    kwargs["const"] = True
            elif (
                hasattr(field.type, "__origin__")
                and re.search(r"^typing\.List\[(.*)\]$", str(field.type)) is not None
            ):
                kwargs["nargs"] = "+"
                kwargs["type"] = partial(none_checker, dtype=field.type.__args__[0])
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), f"{field.name} cannot be a List of mixed types: {field.type.__args__}"
                if field.default_factory is not dataclasses.MISSING:  # type: ignore
                    kwargs["default"] = field.default_factory()  # type: ignore
                elif field.default is dataclasses.MISSING:
                    kwargs["required"] = True
            else:
                kwargs["type"] = partial(none_checker, dtype=field.type)
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:  # type: ignore
                    kwargs["default"] = field.default_factory()  # type: ignore
                else:
                    kwargs["required"] = True
            parser.add_argument(field_name, **kwargs)
