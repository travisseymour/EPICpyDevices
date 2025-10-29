from typing import List

# from epiclibcpp.epiclib import statistics, rms_statistics, geometric_utilities as gu

import epiclibcpp.epiclib

epic_statistics = epiclibcpp.epiclib.statistics
rms_statistics = epiclibcpp.epiclib.rms_statistics
gu = epiclibcpp.epiclib.geometric_utilities


class Mean_accumulator:
    """
    Use these classes to accumulate running statistics values easily.
    reset() - zero the internal variables
    update() - add a new data value, updating current average
    get_n, mean - return the current values
    """

    def __init__(self):
        ...

    def __new__(cls, *args, **kwargs):
        return epic_statistics.Mean_accumulator(*args, **kwargs)

    def get_est_sd(self) -> float:
        ...

    def get_est_var(self) -> float:
        ...

    def get_half_95_ci(self) -> float:
        ...

    def get_mean(self) -> float:
        ...

    def get_n(self) -> int:
        ...

    def get_sample_sd(self) -> float:
        ...

    def get_sample_var(self) -> float:
        ...

    def get_sdm(self) -> float:
        ...

    def get_total(self) -> int:
        ...

    def reset(self):
        ...

    def update(self, x: float):
        ...


class Proportion_accumulator:
    """
    Use these classes to accumulate running statistics values easily.
    reset() - zero the internal variables
    update() - add a new data value, updating current average
    get_n, mean - return the current values
    """

    def __init__(self):
        ...

    def __new__(cls, *args, **kwargs):
        return epic_statistics.Proportion_accumulator(*args, **kwargs)

    def get_count(self) -> int:
        ...

    def get_n(self) -> int:
        ...

    def get_proportion(self) -> float:
        ...

    def reset(self):
        ...

    def update(self, count_it: bool):
        ...


class Distribution_accumulator:
    """
    Accumulate data values into bins and provide the proportion in each bin.
    Initialize with the number of bins and the size in each bin.
    The first bin starts at 0.  Values too big or too small are accumulated
    in the smallest or largest bin.
    Output for( i = 0; i < n_bins; i++) then i*bin_size is the upper bound on values
    in that bin so for bins 0, 25, 50, count for first is number < 0; second is
    number >= 0 and < 25, third is >= 25 and < 50
    """

    def __init__(self, n_bins: int, bin_size: float):
        ...

    def __new__(cls, *args, **kwargs):
        return epic_statistics.Distribution_accumulator(*args, **kwargs)

    def add_counts(self, other: epic_statistics.Distribution_accumulator):
        ...

    def get_bin_count(self, _bin: int) -> int:
        ...

    def get_bin_proportion(self, _bin: int) -> float:
        ...

    def get_bin_size(self) -> float:
        ...

    def get_distribution(self) -> List[float]:
        ...

    def get_max(self) -> float:
        ...

    def get_min(self) -> float:
        ...

    def get_n(self) -> int:
        ...

    def get_n_bins(self) -> int:
        ...

    def reset(self):
        ...

    def update(self, x: float):
        ...


class Correl_accumulator:
    """
    Accumulate data for a correlation coefficient and regression line
    Like the others, this class uses the one-pass approach which
    can be numerically unreliable under some conditions
    """

    def __init__(self):
        ...

    def __new__(cls, *args, **kwargs):
        return epic_statistics.Correl_accumulator(*args, **kwargs)

    def get_intercept(self) -> float:
        ...

    def get_n(self) -> int:
        ...

    def get_r(self) -> float:
        ...

    def get_rsq(self) -> float:
        ...

    def get_slope(self) -> float:
        ...

    def reset(self):
        ...

    def update(self, x: float, y: float):
        ...


class PredObs_accumulator:
    """
    Give this class object a series of predicted and observed values,
    and then get the goodness-of-fit metrics for them
    using regression fit and simple average absolute error
    """

    def __init__(self):
        ...

    def __new__(cls, *args, **kwargs):
        return epic_statistics.PredObs_accumulator(*args, **kwargs)

    def get_intercept(self) -> float:
        ...

    def get_n(self) -> int:
        ...

    def get_rmse(self) -> float:
        ...

    def get_rsq(self) -> float:
        ...

    def get_slope(self) -> float:
        ...

    def reset(self):
        ...

    def update(self, predicted: float, observed: float):
        ...


class Accumulate_rms_error:
    def __init__(self):
        ...

    def __new__(cls, *args, **kwargs):
        return rms_statistics.Accumulate_rms_error(*args, **kwargs)

    def reset(self):
        ...

    def update(self, p1: gu.Point, p2: gu.Point):
        ...

    def get_n(self) -> int:
        ...

    def get_rms(self):
        ...


if __name__ == "__main__":
    import random

    mean_accumulator = Mean_accumulator()
    for i in range(10):
        mean_accumulator.update(random.random() * 100)

    print(f"{mean_accumulator.get_n()=}")
    print(f"{mean_accumulator.get_mean()=}")
    print(f"{mean_accumulator.get_est_var()=}")
