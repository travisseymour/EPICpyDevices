from typing import List, Union
from plum import dispatch

from epiclibcpp.epiclib import Symbol as _Symbol, geometric_utilities as gu


class Symbol:
    """
    Symbol objects hold a character string value and/or a numeric value.
    These values can be replaced, but not modified inside the Symbol.
    If a string value is present, it is used as the basis for comparison.
    The numeric value is compared only if no string value is present.

    The numeric value is either a single double value, a Point, or a vector of Points.
    The single value is stored in both the x and y values of a Point.
    Numeric values are compared using the vector comparisons regardless of whether they
    are a single value, a single Point, or a vector of Points.

    If both string and numeric values are present, the Symbol serves as a "named value"
    especially useful if the value is a vector of many Points.

    The strings and numeric values are stored as unique reference counted objects.
    New values are compared to the stored values, and re-used if they match.

    Creating, copying, assigning, destroying increment/decrement the reference count for
    a value; if the reference count goes to zero, the string or number object id deleted.

    The whole goal of this class is to enable copy & assignment to be done with shallow
    copy and equality comparison to be done with only pointer comparison. Ordering
    comparison is slowed down by the indirection. Since the strings and numeric value
    cannot be modified, no time is wasted on copying values, but the lookup for previous
    matching values slows down creation of new values.

    When the symbol is initialized or assigned from a non-symbol string, a check can made
    about whether the string can be parsed completely as a number (using std::strtod).
    If it can, then the symbol is set up as containing a numeric value only. If not,
    it is set up as containing a string.

    Any attempt to extract a value from a Symbol that does not correspond to the stored
    value results in a Symbol_exception being thrown.

    Extracting a std::string always succeeds - the resulting string contains either the
    string value of the symbol or the numeric value as written with the output operator.
    """

    @dispatch
    def __init__(self):
        ...

    @dispatch
    def __init__(self, s: str, check_for_number: bool = True):
        ...

    @dispatch
    def __init__(self, x: Union[float, int]):
        ...

    @dispatch
    def __init__(self, x: float, y: float):
        ...

    @dispatch
    def __init__(self, p: gu.Point):
        ...

    @dispatch
    def __init__(self, v: List[gu.Point]):
        ...

    @dispatch
    def __init__(self, c: str, v: List[gu.Point]):
        ...

    @dispatch
    def __init__(self, src: _Symbol):
        ...

    def __new__(cls, *args, **kwargs):
        return _Symbol(*args)  # , **kwargs)

    def swap(self, other: _Symbol):
        """swap the members"""
        ...

    def has_string_value(self) -> bool:
        ...

    def has_numeric_value(self) -> bool:
        ...

    def has_single_numeric_value(self) -> bool:
        ...

    def has_point_numeric_value(self) -> bool:
        ...

    def has_pair_numeric_value(self) -> bool:
        ...

    def has_multiple_numeric_value(self) -> bool:
        ...

    def c_str(self) -> str:
        """string value must be present"""
        ...

    def size(self) -> int:
        """string value must be present"""
        ...

    def length(self) -> int:
        """string value must be present"""
        ...

    def str(self) -> str:
        """string value must be present"""
        ...

    def get_numeric_value(self) -> float:
        """must be a single double value"""
        ...

    def get_point(self) -> gu.Point:
        """must be a single Point"""
        ...

    def get_x(self) -> float:
        """must be a single Point present"""
        ...

    def get_y(self) -> float:
        """must be a single Point present"""
        ...

    def get_Point_vector(self) -> List[gu.Point]:
        """get the vector of Points. Must be any numeric value"""
        ...


if __name__ == "__main__":
    sym1 = Symbol("sym1")
    sym2 = Symbol("32.3", True)
    print(f"{sym1.has_string_value()=}")
    print(f"{sym2.has_string_value()=}")
    print(f"{sym1=}")
    print(f"{sym2=}")
    sym3 = Symbol(gu.Point(32, 44))
    print(f"{sym3=}")
    print(f"{sym3.get_point()=}")
    p = gu.Point(402, 1123)
    print(f"{p=}")
