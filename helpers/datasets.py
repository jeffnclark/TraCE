import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import make_blobs


def create_blobs(num=300, seed=10, noise=0.15, centre=(0.75, 0.75)):
    np.random.seed(seed + 1)
    X, y = make_blobs(n_samples=num, n_features=2, centers=[centre, (0.5, 0.25), (0.25, 0.75)], cluster_std=noise)
    X = preprocessing.minmax_scale(X)

    df = pd.DataFrame(preprocessing.minmax_scale(X), columns=["x1", "x2"])
    df["y"] = y - 1
    return df