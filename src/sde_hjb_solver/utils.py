from typing import Generator, Tuple


def get_time_in_hms(dt: float) -> Tuple[int, int, float]:
    """Convert elapsed time in seconds to (hours, minutes, seconds).

    Args:
        dt: Elapsed time in seconds.

    Returns:
        Tuple of (hours, minutes, seconds).
    """
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s


def arange_generator(m: int) -> Generator[int, None, None]:
    """Generate integers from 0 to m - 1.

    Args:
        m: Non-negative upper bound.

    Yields:
        Integers from 0 to m - 1.
    """
    assert type(m) == int, 'm must be an int'
    assert m >= 0, 'm must be non-negative'

    n = 0
    while n < m:
        yield n
        n += 1
