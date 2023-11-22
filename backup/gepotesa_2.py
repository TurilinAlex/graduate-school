import random

import matplotlib.pyplot as plt
import numpy as np

from backup.extremum import Coincident, min_extremum, max_extremum

random.seed(1234)


def min_extremum_1(index: np.ndarray[np.uint32]):
    n = len(index)
    im = np.zeros(shape=(len(index), len(index)), dtype=np.int32)
    for i in range(n):
        for j in range(1, i + 1):
            im[i][i - j] = abs(index[i] - index[i - j])

    return im


def main():
    a = np.array(
        [
            random.randint(-100, 100) +
            (-1) ** random.randint(0, 10) *
            random.randint(-100, 100)
            for _ in range(50)
        ]
    )

    coincident = Coincident()
    e = np.argsort(a=a, kind="mergesort")
    print(e)
    print()
    min_index, min_eps = coincident(min_extremum, index=e, max_coincident=2, eps=1)
    max_index, max_eps = coincident(max_extremum, index=e, max_coincident=2, eps=1)

    ee1 = min_extremum(index=e, eps=2)
    ee2 = min_extremum(index=e, eps=3)
    ee3 = min_extremum(index=e, eps=4)
    ee = min_extremum(index=e, eps=5)
    print(2, ee1)
    print(3, ee2)
    print(4, ee3)
    print(5, ee)
    print()
    ee = np.sort(ee)
    im = min_extremum_1(index=e)
    im = np.sort(a=im, axis=1)

    # find the first non-zero element of each row
    first_nonzero = np.zeros(im.shape[0])
    for i, row in enumerate(im):
        if len(row[np.nonzero(row)]):
            first_nonzero[i] = row[np.nonzero(row)][0]

    # sort the rows based on the first non-zero element
    indices = np.lexsort((first_nonzero,))
    im = im[indices]

    for i, row in enumerate(im):
        print(f"{e[indices[i]]:2d}", end=": ")
        for elem in row:
            if elem != 0:
                print(f"{elem:3d}", end=" ")
        print()

    print(np.all(min_index == ee))
    plt.plot(a)
    plt.scatter(min_index, a[min_index], s=30, c="red", label=f"Eps min = {min_eps}")
    plt.scatter(max_index, a[max_index], s=30, c="green", label=f"Eps max = {max_eps}")
    plt.legend()
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    main()
