import epicpy.epiclib.epiclib.random_utilities as ru
import random

# NOTE: Unfortunately, the seed somehow gets set on pybind11 translation(?), the
#       effect is that all results are now static as if a seed was specified.
#       Instead, I'm just calling the python version when possible
from math import log, exp


def set_random_number_generator_seed(seed: int):
    """
    This might be useful for altering how EPICpy handles some things.
    BUT, if your goal is to alter how the random functions here operate, then just use the
    corresponding call from Python's random module.
    """
    return ru.set_random_number_generator_seed(seed)


""" Random variable generation """


def random_int(rand_range: int) -> int:
    """Returns a random integer in the range 0 and rand_range - 1 inclusive"""
    # return ru.random_int(rand_range)
    return random.randint(0, rand_range - 1)


def biased_coin_flip(p: float) -> bool:
    """Returns True with probability p"""
    # return ru.biased_coin_flip(p)
    return random.random() >= p


def unit_uniform_random_variable() -> float:
    # return ru.unit_uniform_random_variable()
    return random.uniform(0.0, 1.0)


def uniform_random_variable(mean: float, deviation: float) -> float:
    """return a random variable that is uniformly distributed on each side of the
    mean +/- the deviation"""
    # return ru.uniform_random_variable(mean, deviation)
    return 2.0 * deviation * unit_uniform_random_variable() - deviation + mean


def unit_normal_random_variable() -> float:
    # return ru.unit_normal_random_variable()
    return random.uniform(0.0, 1.0)


def normal_random_variable(mean: float, sd: float) -> float:
    """do not call this function if sd == 0"""
    # return ru.normal_random_variable(mean, sd)
    return random.normalvariate(mean, sd)


def exponential_random_variable(theta: float) -> float:
    # return ru.exponential_random_variable(theta)
    return -theta * log(unit_uniform_random_variable())


# def floored_exponential_random_variable(theta: float, floor: float) -> float:
#     return ru.floored_exponential_random_variable(theta, floor)


# def gamma_random_variable(theta: float, n: int) -> float:
#     return ru.gamma_random_variable(theta, n)


def log_normal_random_variable(m: float, s: float) -> float:
    # return ru.log_normal_random_variable(m, s)
    return m * exp(s * unit_normal_random_variable())


def uniform_detection_function(p: float) -> bool:
    """Returns True with a probability = p"""
    # return ru.uniform_detection_function(p)

    rv: float = unit_uniform_random_variable()
    return rv <= p


def gaussian_detection_function(x: float, mean: float, sd: float) -> bool:
    """As x increases, the probability that True is returned increases according to
    a Normal dbn from 0. to 1.0."""
    # return ru.gaussian_detection_function(x, mean, sd)

    threshold: float = (x - mean) / sd  # compute z-score
    rv: float = unit_normal_random_variable()
    return rv <= threshold


def lapsed_gaussian_detection_function(
    x: float, mean: float, sd: float, lapse_probability: float
) -> bool:
    """
    With lapse_probability, return False else return the gaussian_detection_function
    result.
    """
    # return ru.lapsed_gaussian_detection_function(x, mean, sd, lapse_probability)
    if biased_coin_flip(lapse_probability):
        return False
    return gaussian_detection_function(x, mean, sd)


def based_gaussian_detection_function(
    x: float, base: float, mean: float, sd: float
) -> bool:
    """
    As x increases, the probability that True is returned increases according to a
    Normal dbn from base to 1.0.
    """
    # return ru.based_gaussian_detection_function(x, base, mean, sd)

    rv: float = unit_uniform_random_variable()
    if rv <= base:
        return True
    return gaussian_detection_function(x, mean, sd)


def capped_gaussian_detection_function(
    x: float, cap: float, mean: float, sd: float
) -> bool:
    """As x increases, the probability that True is returned increases according to a
    Normal dbn from 0 to cap."""
    # return ru.capped_gaussian_detection_function(x, cap, mean, sd)

    rv: float = unit_uniform_random_variable()
    if rv <= 1.0 - cap:
        return False
    return gaussian_detection_function(x, mean, sd)


def exponential_detection_function(x: float, _lambda: float) -> bool:
    """
    As x increases, the probability that True is returned increases according to an
    exponential dbn from 0 to 1.0 the lambda parameter is assumed to be the
    "small" definition -
    e.g. a value of lambda = .5 gives p(x < 1) = .3868
    cf lambda = 2 giving p(x < 1) = .8706
    """
    # return ru.exponential_detection_function(x, _lambda)

    rv: float = -log(unit_uniform_random_variable()) / _lambda
    return rv <= x


def based_exponential_detection_function(x: float, base: float, _lambda: float) -> bool:
    """As x increases, the probability that True is returned increases according to a
    exponential dbn from base to 1.0."""
    # return ru.based_exponential_detection_function(x, base, _lambda)

    rv: float = unit_uniform_random_variable()
    if rv <= base:
        return True
    return exponential_detection_function(x, _lambda)


def get_bivariate_normal_cdf(z1: float, z2: float) -> float:
    """
    given two z-scores, return the cumulative probability from the bivariate normal.
    # e.g. (-3, +3) returns 0.001348 (+3, +3) returns 0.997302.
    (+1, +2) and (+2, +1) return 0.822204
    """
    return ru.get_bivariate_normal_cdf(z1, z2)


if __name__ == "__main__":
    print(f"{unit_normal_random_variable()}")
    print(f"{biased_coin_flip(.6)}")
    print(f"{random_int(11)}")

    print(f"{unit_normal_random_variable()=}")
    # e.g. (-3, +3) returns 0.001348 (+3, +3) returns 0.997302.
    print(f"{get_bivariate_normal_cdf(-3, 3)=}")
    print(f"{get_bivariate_normal_cdf(3, 3)=}")
