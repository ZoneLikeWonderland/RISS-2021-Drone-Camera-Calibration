import numpy as np
import json
import matplotlib.pyplot as plt


def fit(xs, ys, k=3):
    xs = np.array(xs)
    ys = np.array(ys)
    n = len(ys)
    A = np.ones((n, k+1))
    for i in range(1, k+1):
        A[:, i] = xs**(i*2)

    a, res, rank, sigma = np.linalg.lstsq(A, ys, rcond=None)
    print(a)

    plt.figure()
    plt.scatter(xs, ys)

    xt = np.linspace(0, 0.5)
    yt = np.zeros_like(xt)
    for i in range(0, k+1):
        yt += a[i]*xt**(i*2)
    plt.plot(xt, yt)
    plt.ylim(0, 0.5)
    plt.show()

    return a


if __name__ == "__main__":
    data = json.load(open("data.json"))

    xs = np.array([i[0] for i in data])/300
    ys = np.array([i[1] for i in data])

    fit(xs, ys)
