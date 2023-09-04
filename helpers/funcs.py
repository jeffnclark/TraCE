import numpy as np
import math
from scipy.special import gamma


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


def n_sphere(v):
    r = np.linalg.norm(v)
    n = len(v)
    V = (np.pi ** (n / 2) / (gamma(n / 2 + 1))) * r ** n
    return V


def score(x0, x1, x_prime, func=None):
    """
    return the score of the trajectory as a balance between where we want to go, where we
    don't want to go, and where we actually went
    :param x0: start point
    :param x1: end point
    :param x_prime: counterfactual
    :return: score
    """
    v = vec(x0, x1)
    v_prime0 = vec(x0, x_prime)
    v_prime1 = vec(x1, x_prime)
    length = np.linalg.norm

    phi = np.dot(v_prime0, v) / (length(v_prime0) * length(v))

    if phi == 1:
        psi = length(v) / length(v_prime0)
    elif phi <= 0:
        v_best = v_prime0
        psi = np.dot(v_best, v_prime1) / (length(v_best) * length(v_prime1))
    else:
        mag = np.sqrt(length(v_prime0) ** 2 - (length(v_prime0) * np.sqrt(1 - phi ** 2)) ** 2)
        x_best = x0 + (v / length(v)) * mag
        v_best = x_prime - x_best
        psi = np.dot(v_best, v_prime1) / (length(v_best) * length(v_prime1))

    lam = func(length(v_prime1 - v_prime0))
    S = lam * phi + (1 - lam) * psi

    return S


def score2(x0, x1, x_prime, func=None, weight=0.5):
    """
    return the score of the trajectory as a balance between where we want to go, where we
    don't want to go, and where we actually went
    :param x0: start point
    :param x1: end point
    :param x_prime: counterfactual
    :return: score
    """
    v = vec(x0, x1)
    v_prime0 = vec(x0, x_prime)
    v_prime1 = vec(x1, x_prime)
    length = np.linalg.norm

    if func is None:
        h = 2 / (1 + np.exp(-(n_sphere(v_prime0) - n_sphere(v_prime1)))) - 1
    else:
        h = func(x0, x1, x_prime)

    f = 1 / 2 * (np.dot(v, v_prime0) / (length(v) * length(v_prime0)) +
                 np.dot(v, v_prime1) / (length(v) * length(v_prime1)))
    if f == 0:
        S = h
    else:
        S = weight * f + (1 - weight) * h

    return S


def score1(x0, x1, x_prime, x_star=None, method='dot'):
    """
    return the score of the trajectory as a balance between where we want to go, where we
    don't want to go, and where we actually went
    :param method: method for scoring (dot, avg, interp)
    :param x0: start point
    :param x1: end point
    :param x_prime: positive counterfactual
    :param x_star: negative counterfactual
    :return: score
    """
    if method == 'dot':
        if x_star is None:
            v = vec(x0, x1)
            v_prime = vec(x0, x_prime)
            S = cos_sim(v, v_prime)

        else:
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
            S = np.dot((norm_v_prime - norm_v_star) / length, norm_v)
            # S1 = (np.dot(v, v_prime) / (np.linalg.norm(v) * np.linalg.norm(v_prime))) * (np.linalg.norm(v) / np.linalg.norm(v_prime))
            # S2 = (np.dot(v, v_star) / (np.linalg.norm(v) * np.linalg.norm(v_star))) * (np.linalg.norm(v) / np.linalg.norm(v_star))
            # S = (S1 - S2) / 2

    if method == 'avg':
        S = 0
        v = vec(x0, x1)
        norm_v = v / np.linalg.norm(v)
        n = 0
        for xp in x_prime:
            v_prime = vec(x0, xp)
            p_test1 = (np.dot(v, v_prime) / (np.linalg.norm(v_prime) * np.linalg.norm(v)))
            p_test2 = (1 - np.exp(-np.linalg.norm(v)))
            p_test3 = (np.exp(-np.linalg.norm(x1 - xp)))
            temp = (np.dot(v, v_prime) / (np.linalg.norm(v_prime) * np.linalg.norm(v))) * (
                    1 - np.exp(-np.linalg.norm(v))) * (np.exp(-np.linalg.norm(x1 - xp)))
            S += temp
            n += 1
        for xs in x_star:
            v_star = vec(x0, xs)
            s_test1 = (np.dot(v, v_star) / (np.linalg.norm(v_star) * np.linalg.norm(v)))
            s_test2 = (1 - np.exp(-np.linalg.norm(v)))
            s_test3 = (np.exp(-np.linalg.norm(x1 - xs)))
            temp = (np.dot(v, v_star) / (np.linalg.norm(v_star) * np.linalg.norm(v))) * (
                    1 - np.exp(-np.linalg.norm(v))) * (np.exp(-np.linalg.norm(x1 - xs)))
            S -= temp
            n += 1

        S = S / n

    if method == 'interp':
        S = 0
        # get vectors as normalised differences
        v = vec(x0, x1)
        v_prime = np.zeros(x_prime.shape)
        for i, prime in enumerate(x_prime):
            temp = vec(x0, prime)
            v_prime[i] = temp / np.linalg.norm(temp)
        v_star = np.zeros(x_star.shape)
        for i, star in enumerate(x_star):
            temp = vec(x0, star)
            v_star[i] = temp / np.linalg.norm(temp)

        # convert to polar

    return S


def news(x):
    """
    Calculates the NEWS score, whislt capturing the good counterfactual and bad counterfactual in parallel
    :param x: vector of NEWS values (resp. rate, ox. sat., ox. sup., temp., blood pres., hrt rt, consciousness)
    :return: news score, good counterfactual x_prime, bad counterfactual x_star
    """
    # set up values for calculation
    score = 0
    x_prime = np.zeros(len(x))
    x_star = np.zeros(len(x))
    # perfect values from mean of 0 score
    x_prime[0] = 16  # respiration rate
    x_prime[1] = 100  # oxygen saturation levels
    x_prime[2] = 0  # any supplemental oxygen
    x_prime[3] = 37.05  # temperature
    x_prime[4] = 165  # blood pressure
    x_prime[5] = 70.5  # heart rate
    x_prime[6] = 0  # consciousness

    # calculate respiratory score
    if x[0] <= x_prime[0]:
        x_star[0] = 12
    else:
        x_star[0] = 20
    if x[0] <= 11:
        score += 1
        x_star[0] = 8
    if x[0] <= 8:
        score += 2
        x_star[0] = x[0] - 1
    if x[0] >= 21:
        score += 2
        x_star[0] = 25
    if x[0] >= 25:
        score += 1
        x_star[0] = x[0]

    # calculate oxygen saturation score
    if x[1] >= 96:
        x_star[1] = 95
    if x[1] <= 95:
        score += 1
        x_star[1] = 93
    if x[1] <= 93:
        score += 1
        x_star[1] = 91
    if x[1] <= 91:
        score += 1
        x_star[1] = x[1]

    # calculate supplemental oxygen score (1 on supplementary oxygen, 0 not)
    x_star[2] = 1
    if x[2] == 1:
        score += 2

    # calculate temperature score
    if x[3] <= x_prime[3]:
        x_star[3] = 36
    else:
        x_star[3] = 38.1
    if x[3] <= 36:
        score += 1
        x_star[3] = 35
    if x[3] <= 35:
        score += 2
        x_star[3] = x[3] - 1
    if x[3] >= 38.1:
        score += 1
        x_star[3] = 39.1
    if x[3] >= 39.1:
        score += 1
        x_star[3] = x[3]

    # calculate blood pressure score
    if x[4] <= x_prime[4]:
        x_star[4] = 110
    else:
        x_star[4] = 220
    if x[4] <= 110:
        score += 1
        x_star[4] = 100
    if x[4] <= 100:
        score += 1
        x_star[4] = 90
    if x[4] <= 90:
        score += 1
        x_star[4] = x[4]
    if x[4] >= 220:
        score += 3
        x_star[4] = x[4]

    # calculate heart rate score
    if x[5] <= x_prime[5]:
        x_star[5] = 50
    else:
        x_star[5] = 91
    if x[5] <= 50:
        score += 1
        x_star[5] = 40
    if x[5] <= 40:
        score += 1
        x_star[5] = x[5]
    if x[5] >= 91:
        score += 1
        x_star[5] = 111
    if x[5] >= 111:
        score += 1
        x_star[5] = 131
    if x[5] >= 131:
        score += 1
        x_star[5] = x[0]

    # calculate level of consciousness score (1 unconscious, 0 conscious)
    x_star[6] = 1
    if x[6] == 1:
        score += 3

    return score, x_prime, x_star
