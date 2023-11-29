import time

import numpy as np
from TradingMath.extremal import argsort


def main():
    size = 1_000_000
    data = np.random.uniform(0, 5000000, size)

    index = np.empty_like(data, dtype=np.int64)
    s = time.monotonic()
    argsort(data, index)
    e = time.monotonic()
    print(e - s)
    s = time.monotonic()
    np.argsort(data, kind="mergesort")
    e = time.monotonic()
    print(e - s)


if __name__ == "__main__":
    main()
