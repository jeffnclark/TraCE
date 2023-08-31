import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import make_blobs


def create_blobs(num=300, seed=10, noise=0.15, centre=(0.75, 0.75)):
    '''
    Generate simple 2D dataset

    Args:

    Returns:
        df: dataframe of generated points

    '''
    np.random.seed(seed + 1)
    X, y = make_blobs(n_samples=num, n_features=2, centers=[
        centre, (0.5, 0.25), (0.25, 0.75)], cluster_std=noise)
    X = preprocessing.minmax_scale(X)

    dataset_df = pd.DataFrame(
        preprocessing.minmax_scale(X), columns=["x1", "x2"])
    dataset_df["y"] = y - 1
    return dataset_df


def load_dataset(dataset_name):
    '''
    Load dataset from csv

    Arguments:
        dataset_name (str): Dataset name

    Returns
        X: Dataframe of input features
        y: Numpy array of label
    '''

    current_path = os.path.dirname(os.path.realpath(__file__))

    # imports the unscaled version
    if dataset_name == 'mimic_original':
        dataset_file_path = f"{current_path}/fm_MIMIC_COMPLETECASE_original.csv"
        dataset_df = pd.read_csv(dataset_file_path, header=0, engine="python")

    # or the scaled version
    elif dataset_name == 'mimic_scaled':
        dataset_file_path = f"{current_path}/fm_MIMIC_COMPLETECASE_original_STANDARDISED.csv"
        dataset_df = pd.read_csv(dataset_file_path, header=0, engine="python")

    # drop unncessary cohort column
    dataset_df.drop(['cohort'], axis=1, inplace=True)

    # rename outcome as y
    dataset_df.rename(columns={"outcome": "y"}, inplace=True)

    return dataset_df


def generate_expired(dataset_name, expired_fraction=0.05):
    '''
    Add a small number of expired (in-hospital mortality) labels to the currently binary task
    Randomly sample and create new label, -1 = expired
    '''
    dataset_df = load_dataset(dataset_name)

    dfupdate = dataset_df.sample(frac=expired_fraction)
    dfupdate.y = -1
    dataset_df.update(dfupdate)

    return dataset_df


def normalise_news(x):
    temp = np.copy(x)
    temp[0] = (x[0] - 8) / (25 - 8)
    temp[1] = (x[1] - 91) / (100 - 91)
    temp[3] = (x[3] - 35) / (39.1 - 35)
    temp[4] = (x[4] - 90) / (220 - 90)
    temp[5] = (x[5] - 40) / (131 - 40)
    return x
