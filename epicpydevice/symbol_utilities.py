from typing import List

from plum import dispatch

from epiclibcpp.epiclib import (
    Output_tee,
    Symbol,
    symbol_utilities,
    geometric_utilities as gu,
)


# Return the Symbol that is nth in the list, starting with 0
# if not found, or not legal n, an empty Symbol is returned
def get_nth_Symbol(in_list: List[Symbol], n: int) -> Symbol:
    return symbol_utilities.get_nth_Symbol(in_list, n)


@dispatch
def print_Symbol_list(in_list: List[Symbol]):
    for symbol in in_list:
        print(symbol)


@dispatch
def print_Symbol_list(in_list: List[Symbol], ot: Output_tee):
    for symbol in in_list:
        ot(symbol)


def cstr_to_Symbol_list(text: str) -> List[Symbol]:
    """
    return a list of Symbols, where each symbol is the whitespace delimited sequence
    in the input C-string.  e.g. "A B CD E" => (A B CD E)
    """
    return [Symbol(item) for item in text.split(" ")]


def int_to_Symbol(i: int) -> Symbol:
    return symbol_utilities.int_to_Symbol(i)


@dispatch
def concatenate_to_Symbol(text: str, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(text, i)


@dispatch
def concatenate_to_Symbol(sym: Symbol, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(sym, i)


@dispatch
def concatenate_to_Symbol(text1: str, text2: str, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(text1, text2, i)


@dispatch
def concatenate_to_Symbol(sym1: Symbol, sym2: Symbol, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(sym1, sym2, i)


@dispatch
def concatenate_to_Symbol(text: str, sym: Symbol, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(text, sym, i)


@dispatch
def concatenate_to_Symbol(sym: Symbol, text: str, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(sym, text, i)


@dispatch
def concatenate_to_Symbol(text1: str, text2: str, text3: str, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(text1, text2, text3, i)


@dispatch
def concatenate_to_Symbol(text1: str, sym: Symbol, text3: str, i: int) -> Symbol:
    """
    return a Symbol consisting of the supplied string(s) or Symbol(s)
    followed by the digits of the supplied integer
    """
    return symbol_utilities.concatenate_to_Symbol(text1, sym, text3, i)


def concatenate_to_String(symbols: List[Symbol]) -> str:
    return symbol_utilities.concatenate_to_String(symbols)


if __name__ == "__main__":
    sym1 = Symbol("sym1")
    sym2 = Symbol("32", True)
    sym3 = Symbol(gu.Point(32, 44))
    sym4 = concatenate_to_Symbol("one", "two", "three", 4)
    sym5 = int_to_Symbol(1138)
    symbols = list((sym1, sym2, sym3, sym4, sym5))
    print(f"{symbols=}")
    print(f"{get_nth_Symbol(symbols, 4)=}")
