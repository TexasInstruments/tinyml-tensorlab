"""Regression test for str2int_or_float()'s unreachable float branch.

not v.isdigit() is True for any string containing '.' (e.g. '31.25') or a
leading '-', so checking it before the '.' in v branch made float conversion
dead code -- every float-like string fell through to the string passthrough
instead of being converted, silently breaking any caller (e.g. Downsample())
that does arithmetic assuming a numeric type.
"""
from tinyml_tinyverse.common.utils.misc_utils import str2int_or_float


def test_str2int_or_float_converts_fractional_strings():
    assert str2int_or_float("31.25") == 31.25
    assert isinstance(str2int_or_float("31.25"), float)


def test_str2int_or_float_converts_plain_integer_strings():
    assert str2int_or_float("42") == 42
    assert isinstance(str2int_or_float("42"), int)


def test_str2int_or_float_passes_through_non_numeric_strings():
    assert str2int_or_float("abc") == "abc"


def test_str2int_or_float_passes_through_non_numeric_strings_with_a_dot():
    assert str2int_or_float("abc.def") == "abc.def"


def test_str2int_or_float_handles_special_string_sentinels():
    assert str2int_or_float("None") is None
    assert str2int_or_float("True") is True
    assert str2int_or_float("False") is False


def test_str2int_or_float_passes_through_non_string_values():
    assert str2int_or_float(5) == 5
    assert str2int_or_float(5.5) == 5.5
    assert str2int_or_float(None) is None
