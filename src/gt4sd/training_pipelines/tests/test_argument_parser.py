"""Argument parser unit tests."""

from dataclasses import dataclass, field
from typing import Union

from gt4sd.cli.argument_parser import ArgumentParser


@dataclass
class TestArguments:

    int_arg: int = field(default=0)

    float_arg: float = field(default=0.0)

    str_arg: str = field(default="test")

    bool_arg: bool = field(default=True)

    int_none_arg: Union[int, None] = field(default=None)

    float_none_arg: Union[float, None] = field(default=None)

    str_none_arg: Union[str, None] = field(default=None)

    bool_none_arg: Union[bool, None] = field(default=None)


def test_int_default():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert isinstance(args[0].int_arg, int)
    assert args[0].int_arg == 0


def test_float_default():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert isinstance(args[0].float_arg, float)
    assert args[0].float_arg == 0.0


def test_str_default():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert isinstance(args[0].str_arg, str)
    assert args[0].str_arg == "test"


def test_bool_default():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert isinstance(args[0].bool_arg, bool)
    assert args[0].bool_arg is True


def test_int_assigned():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--int_arg", "1"])

    assert isinstance(args[0].int_arg, int)
    assert args[0].int_arg == 1


def test_float_assigned():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--float_arg", "1.0"])

    assert isinstance(args[0].float_arg, float)
    assert args[0].float_arg == 1.0


def test_str_assigned():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--str_arg", "my_test"])

    assert isinstance(args[0].str_arg, str)
    assert args[0].str_arg == "my_test"


def test_bool_assigned():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--bool_arg", "False"])

    assert isinstance(args[0].bool_arg, bool)
    assert args[0].bool_arg is False


def test_bool_int_assigned():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--bool_arg", "0"])

    assert isinstance(args[0].bool_arg, bool)
    assert args[0].bool_arg is False


def test_int_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert args[0].int_none_arg is None


def test_float_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert args[0].float_none_arg is None


def test_str_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert args[0].str_none_arg is None


def test_bool_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses([])

    assert args[0].bool_none_arg is None


def test_int_str_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--int_none_arg", ""])

    assert args[0].int_none_arg is None


def test_float_str_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--float_none_arg", ""])

    assert args[0].float_none_arg is None


def test_str_str_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--str_none_arg", ""])

    assert args[0].str_none_arg is None


def test_bool_str_none():

    parser = ArgumentParser((TestArguments))  # type: ignore

    args = parser.parse_args_into_dataclasses(["--bool_none_arg", ""])

    assert args[0].bool_none_arg is None
