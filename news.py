import numpy as np
import matplotlib.pyplot as plt
from helpers.funcs import *
from data.datasets import normalise_news

# patient at t = 0
x0 = np.zeros(7)
x0[0] = 8  # respiration rate
x0[1] = 91  # oxygen saturation levels
x0[2] = 1  # any supplemental oxygen
x0[3] = 38.5  # temperature
x0[4] = 95  # blood pressure
x0[5] = 48  # heart rate
x0[6] = 0 # consciousness

# patient at t = 1
x1 = np.zeros(7)
x1[0] = 8  # respiration rate
x1[1] = 92  # oxygen saturation levels
x1[2] = 1  # any supplemental oxygen
x1[3] = 38.5  # temperature
x1[4] = 95  # blood pressure
x1[5] = 46  # heart rate
x1[6] = 0 # consciousness

# patient at t = 2
x2 = np.zeros(7)
x2[0] = 9  # respiration rate
x2[1] = 95  # oxygen saturation levels
x2[2] = 1  # any supplemental oxygen
x2[3] = 38  # temperature
x2[4] = 110  # blood pressure
x2[5] = 49  # heart rate
x2[6] = 0 # consciousness

# patient at t = 3 (perfect health)
x3 = np.zeros(7)
x3[0] = 12  # respiration rate
x3[1] = 96  # oxygen saturation levels
x3[2] = 0  # any supplemental oxygen
x3[3] = 37.9  # temperature
x3[4] = 130  # blood pressure
x3[5] = 55  # heart rate
x3[6] = 0 # consciousness

# get vectors and news score
score0, prime0, star0 = news(x0)
score1, prime1, star1 = news(x1)
score2, prime2, star2 = news(x2)
score3, prime3, star3 = news(x3)

# normalise everything and get TraCE scores
R1 = score(x0, x1, prime0, star0)
R2 = score(x1, x2, prime1, star1)
R3 = score(x2, x3, prime2, star2)

print('At t=0:')
print('patient values: {}'.format(x0))
print('NEWS Score: {}'.format(score0))
print('----------------------------------------------')
print('At t=1')
print('patient values: {}'.format(x1))
print('NEWS Score: {}'.format(score1))
print('desired direction: {}'. format(prime0 - x0))
print('undesired direction: {}'.format(star0 - x0))
print('actual direction: {}'.format(x1 - x0))
print('TraCE Score: {}'.format(R1))
print('----------------------------------------------')
print('At t=2')
print('patient values: {}'.format(x2))
print('NEWS Score at: {}'.format(score2))
print('desired direction: {}'. format(prime1 - x1))
print('undesired direction: {}'.format(star1 - x1))
print('actual direction: {}'.format(x2 - x1))
print('TraCE Score at: {}'.format(R2))
print('At t=3')
print('patient values: {}'.format(x3))
print('NEWS Score at: {}'.format(score3))
print('desired direction: {}'. format(prime2 - x2))
print('undesired direction: {}'.format(star2 - x2))
print('actual direction: {}'.format(x3 - x2))
print('TraCE Score at: {}'.format(R3))
