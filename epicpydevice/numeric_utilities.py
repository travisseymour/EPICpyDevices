from epiclibcpp.epiclib import numeric_utilities as nu

from plum import dispatch


def round_to_integer(x: float) -> float:
    """rounds x to an integer value returned in a double"""
    return nu.round_to_integer(x)


def int_to_string(i: int) -> str:
    """writes an integer value to a string"""
    return nu.int_to_string(i)


@dispatch
def time_convert(hours: int, minutes: int, seconds: int) -> int:
    """convert hours, minutes, and seconds to milliseconds
    (long integer returned: note)"""
    return nu.time_convert(hours, minutes, seconds)


@dispatch
def time_convert(
    time_ms: int, hours: int, minutes: int, seconds: int, milliseconds: int
):
    """convert milliseconds into hours, minutes, seconds, and milliseconds"""
    return nu.time_convert(time_ms, hours, minutes, seconds, milliseconds)


@dispatch
def time_convert(time_ms: int, hours: int, minutes: int, seconds: float):
    """convert milliseconds into hours, minutes, and double seconds"""
    return nu.time_convert(time_ms, hours, minutes, seconds)


def logb2(x: float) -> float:
    """compute the base 2 log of the supplied value"""
    return nu.logb2(x)


def pitch_to_semitones(pitch: float) -> float:
    return nu.pitch_to_semitones(pitch)


if __name__ == "__main__":
    print(f"{round_to_integer(3.14)}")
    print(f"{int_to_string(42)}")
    print(f"{pitch_to_semitones(440)}")
