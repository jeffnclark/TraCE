import numpy as np


def vec(x1, x2):
    """
    Create the vector that defines the linear movement between two points
    :param x1: start point
    :param x2: end point
    :return: vector between points
    """
    return x2 - x1


def cos_sim(v, u):
    """
    calculate the normalised angle between two vectors.
    u and v must be vectors of size n
    :param v: vector 1
    :param u: vector 2
    :return: similarity [-1, 1]
    """
    v_len = np.linalg.norm(v)
    u_len = np.linalg.norm(u)
    dot = np.dot(v, u)
    return dot / (v_len * u_len)


def score(x0, x1, x_prime, x_star):
    """
    return the score of the trajectory as a balance between where we want to go, where we
    don't want to go, and where we actually went
    :param x0: start point
    :param x1: end point
    :param x_prime: positive counterfactual
    :param x_star: negative counterfactual
    :return: score
    """
    # get vectors
    v = vec(x0, x1)
    v_prime = vec(x0, x_prime)
    v_star = vec(x0, x_star)

    # get angle
    theta = cos_sim(v_prime, v_star)

    if theta == 1:
        print('Counterfactuals aligned, cannot calculate score')
        return np.nan

    # scale score by maximum possible potential score
    norm_v_prime = v_prime / np.linalg.norm(v_prime)
    norm_v_star = v_star / np.linalg.norm(v_star)
    norm_v = v / np.linalg.norm(v)
    length = np.sqrt(2 - 2 * theta)
    score = np.dot((norm_v_prime - norm_v_star) / length, norm_v)

    return score

