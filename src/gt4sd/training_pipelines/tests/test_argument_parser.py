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
