import numpy as np
import matplotlib.pyplot as plt
from helpers.funcs import *
from data.datasets import normalise_news

# an example of when NEWS can be misleading
# patient at t = 0 (reasonably healthy patient)
x0 = np.zeros(7)
x0[0] = 14  # respiration rate
x0[1] = 97  # oxygen saturation levels
x0[2] = 0  # any supplemental oxygen
x0[3] = 37.8  # temperature
x0[4] = 106  # blood pressure
x0[5] = 49  # heart rate
x0[6] = 0  # consciousness

# patient at t = 1 (patient goes critical, but near boundaries)
x1 = np.zeros(7)
x1[0] = 9  # respiration rate
x1[1] = 94  # oxygen saturation levels
x1[2] = 0  # any supplemental oxygen
x1[3] = 39  # temperature
x1[4] = 91  # blood pressure
x1[5] = 41  # heart rate
x1[6] = 0  # consciousness

# patient at t = 2 (mostly improves, but one variable crosses a worse boundary)
x2 = np.zeros(7)
x2[0] = 10  # respiration rate
x2[1] = 93  # oxygen saturation levels
x2[2] = 0  # any supplemental oxygen
x2[3] = 38.5  # temperature
x2[4] = 95  # blood pressure
x2[5] = 45  # heart rate
x2[6] = 0  # consciousness

# patient at t = 3 (Stays stable score-wise, but most variables improve)
x3 = np.zeros(7)
x3[0] = 11  # respiration rate
x3[1] = 93  # oxygen saturation levels
x3[2] = 0  # any supplemental oxygen
x3[3] = 38.1  # temperature
x3[4] = 100  # blood pressure
x3[5] = 50  # heart rate
x3[6] = 0  # consciousness

# patient at t = 4 (perfect health)
x4 = np.zeros(7)
x4[0] = 12  # respiration rate
x4[1] = 96  # oxygen saturation levels
x4[2] = 0  # any supplemental oxygen
x4[3] = 37.9  # temperature
x4[4] = 130  # blood pressure
x4[5] = 55  # heart rate
x4[6] = 0  # consciousness

# get vectors and news score
score0, prime0, star0 = news(x0)
score1, prime1, star1 = news(x1)
score2, prime2, star2 = news(x2)
score3, prime3, star3 = news(x3)
score4, prime4, star4 = news(x4)

# normalise everything and get TraCE scores
R1 = np.round(score(x0, x1, [prime0], [star0], method='avg'), 2)
R2 = np.round(score(x1, x2, [prime1], [star1], method='avg'), 2)
R3 = np.round(score(x2, x3, [prime2], [star2], method='avg'), 2)
R4 = np.round(score(x3, x4, [prime3], [star3], method='avg'), 2)

t = np.array([0, 1, 2, 3, 4])
scores = np.array([score0, score1, score2, score3, score4])
R = np.array([R1, R2, R3, R4])

print('At t=0 (patient in reasonable health)')
print('patient values: {}'.format(x0))
print('NEWS Score: {}'.format(score0))
print('----------------------------------------------')
print('At t=1 (critical patient near boundaries)')
print('patient values: {}'.format(x1))
print('NEWS Score: {}'.format(score1))
print('trajectory t=0->1: {}'.format(x1 - x0))
print('TraCE Score: {}'.format(R1))
print('----------------------------------------------')
print('At t=2 (most variables improve, but one crosses a deterioration boundary)')
print('patient values: {}'.format(x2))
print('NEWS Score: {}'.format(score2))
print('trajectory t=1->2: {}'.format(x2 - x1))
print('TraCE Score: {}'.format(R2))
print('----------------------------------------------')
print('At t=3 (most variables improve a tiny bit, but not enough to cross a boundary)')
print('patient values: {}'.format(x3))
print('NEWS Score: {}'.format(score3))
print('trajectory t=2->3:: {}'.format(x3 - x2))
print('TraCE Score: {}'.format(R3))
print('----------------------------------------------')
print('At t=4 (patient in "perfect health")')
print('patient values: {}'.format(x4))
print('NEWS Score: {}'.format(score4))
print('trajectory t=3->4: {}'.format(x4 - x3))
print('TraCE Score: {}'.format(R4))

f, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
y_labels = ['NEWS', 'TraCE']
x_lims = [[0, 8], [-1, 1]]
for i, ax in enumerate(axs):
    ax.set_xlim([0,4])
    ax.set_ylim(x_lims[i])
    ax.set_ylabel(y_labels[i])

axs[0].plot(t, scores)
axs[1].plot(t[1::], R)
axs[1].set_xlabel('time')
plt.show()

print('end')
