"""Argument parser for training pipelines."""


import dataclasses
import re
from argparse import ArgumentTypeError
from enum import Enum
from functools import partial
from typing import Any, List, NewType, Optional, Type, Union

from transformers import HfArgumentParser


def none_checker_bool(val: Union[bool, str]) -> Union[bool, None]:
    """Check given bool argument for None.

    Args:
        val: model arguments passed to the configuration.
    Returns:
        Bool value or None.
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
                ), f"{field.name} cannot be a List of mixed types"
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
