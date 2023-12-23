import time

import numpy as np
from TradingMath.sort import argsort


def main():
    size = 1_000_000
    data = np.random.uniform(0, 5000000, size)

    index_1 = np.empty_like(data, dtype=np.int32)
    s = time.monotonic()
    argsort(data, index_1, parallel=True)
    e = time.monotonic()
    print(e - s)

    index_2 = np.empty_like(data, dtype=np.int32)
    s = time.monotonic()
    argsort(data, index_2, parallel=False)
    e = time.monotonic()
    print(e - s)

    s = time.monotonic()
    index_3 = np.argsort(data, kind="mergesort")
    e = time.monotonic()
    print(e - s)
    print(all(index_1 == index_3))


if __name__ == "__main__":
    main()
