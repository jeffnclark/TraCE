import numpy as np
import data.datasets as datasets
from helpers.plotters import *
from helpers.funcs import *
from sklearn.neural_network import MLPClassifier

# settings
samples = 500
seed = 10
noise = 0.25
plot = False

# define data generator
generator = datasets.create_blobs  # get data generator

# create data set and model for scenario 1
data1 = generator(samples, seed, noise, centre=(0.75, 0.75))  # generate data
x1 = data1.iloc[:, :2]  # ensure data is 2d
y1 = data1.y

model1 = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1, max_iter=700).fit(x1, y1)  # get target
X1 = data1[["x1", "x2"]]

# create data set and model for scenario 2
data2 = generator(samples, seed, noise, centre=(1.3, 0.2))  # generate data
x2 = data2.iloc[:, :2]  # ensure data is 2d
y2 = data2.y

model2 = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1, max_iter=700).fit(x2, y2)  # get target
X2 = data2[["x1", "x2"]]

# factual at t=0, t=1, t=2
factual = np.array([[0.1, 0.1],
                    [0.2, 0.25],
                    [0.125, 0.35]])

factual2 = np.array([[0.3, 0.2],
                    [0.5, 0.3],
                    [0.35, 0.45]])

cf1 = np.array([[[0.1, 0.5],
                 [0.35, 0.6]],
                [[0.025, 0.4],
                 [0.1, 0.45]]
                ])
cf2 = np.array([[[0.6, 0.6],
                 [0.65, 0.5]],
                [[0.625, 0.1],
                 [0.635, 0.25]]
                ])

cf11 = np.array([[[0.25, 0.5],
                 [0.4, 0.55]],
                [[0.175, 0.5],
                 [0.365, 0.575]]
                ])
cf22 = np.array([[[0.625, 0.2],
                 [0.635, 0.3]],
                [[0.625, 0.2],
                 [0.635, 0.3]]
                ])

test_x0 = np.array([0, 0])
test_x1 = np.array([0, 1])
test_xprime = np.array([[1, 0], [-1, 1]])
test_func1 = lambda a: 1 - a / np.sqrt(2)
test_func2 = lambda a: np.exp(-a)
test_func3 = lambda a: 0.5
test = score(test_x0, test_x1, test_xprime[0], func=test_func3)
# scores
scores = np.zeros((2, 2))
for i in range(len(factual) - 1):
    xt = factual[i, :]
    xt1 = factual[i + 1, :]
    for j in range(cf1.shape[2]):
        div = 2
        x_prime = cf1[j, i, :]
        x_star = cf2[j, i, :]
        temp1 = score(xt, xt1, x_prime, func=test_func3)
        temp2 = score(xt, xt1, x_star, func=test_func3)
        temp3 = 1
        temp4 = 1
        if j == 1:
            div = 4
            x_prime = cf1[j-1, i, :]
            x_star = cf2[j-1, i, :]
            temp3 = score(xt, xt1, x_prime, func=test_func3)
            temp4 = score(xt, xt1, x_star, func=test_func3)
        scores[i, j] = 1 / div * (temp1 - temp2 + temp3 - temp4)

scores2 = np.zeros((2, 2))
for i in range(len(factual) - 1):
    xt = factual2[i, :]
    xt1 = factual2[i + 1, :]
    for j in range(cf11.shape[2]):
        div = 2
        x_prime = cf11[j, i, :]
        x_star = cf22[j, i, :]
        temp1 = score(xt, xt1, x_prime, func=test_func3)
        temp2 = score(xt, xt1, x_star, func=test_func3)
        temp3 = 1
        temp4 = 1
        if j == 1:
            div = 4
            x_prime = cf11[j - 1, i, :]
            x_star = cf22[j - 1, i, :]
            temp3 = score(xt, xt1, x_prime, func=test_func3)
            temp4 = score(xt, xt1, x_star, func=test_func3)
        scores2[i, j] = 1 / div * (temp1 - temp2 + temp3 - temp4)

        # scores[i, j] = score(xt, xt1, [x_prime], [x_star], method='avg')

# plotting
if plot:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 10), sharey=True, sharex=True)

    ax[0] = plot_dataset(ax[0], data1)
    ax[0] = plot_decision_boundary(ax[0], X1, model1)

    ax[1] = plot_dataset(ax[1], data2)
    ax[1] = plot_decision_boundary(ax[1], X2, model2)

    for j, ax in enumerate(ax):
        ax.plot(factual[:, 0], factual[:, 1], 'ko-', label='true trajectory')
        ax.set(aspect='equal')
        for i, fact in enumerate(factual):
            ax.text(fact[0] + 0.02, fact[1], r'$x_{}$'.format(i), va='center', ha='center',
                    rotation='horizontal', fontsize=16, color='black', alpha=.7, zorder=100)
        ax.plot(cf1[j, :, 0], cf1[j, :, 1], 'k*', markersize=15, markeredgecolor='white')
        ax.plot(cf2[j, :, 0], cf2[j, :, 1], 'w*', markersize=15, markeredgecolor='black')

        for i in range(len(cf1[j])):
            shift = 0
            if i == 0:
                shift = 0.05
            ax.text(((factual[i, 0] + factual[i + 1, 0]) / 2) + 0.05, (factual[i, 1] + factual[i + 1, 1]) / 2 - shift,
                    r'$S_{} = {}$'.format(i, np.round(scores[i, j], 4)), va='center', ha='center',
                    rotation='horizontal', fontsize=16, color='black', alpha=.7, zorder=100)
            line1 = ax.annotate("", xy=(cf1[j, i, 0], cf1[j, i, 1]), xytext=(factual[i, 0], factual[i, 1]),
                                arrowprops=dict(arrowstyle="->", lw=2, color='blue', alpha=.7))
            line2 = ax.annotate("", xy=(cf2[j, i, 0], cf2[j, i, 1]), xytext=(factual[i, 0], factual[i, 1]),
                                arrowprops=dict(arrowstyle="->", lw=2, color='red', alpha=.7))

        if j == 1:
            ax.legend(loc='upper right', fancybox=True, framealpha=0.2, prop={'size': 14}).set_zorder(1000)
        ax.set_xlim([0, 0.8])
        ax.set_ylim([0, 0.8])

    plt.tight_layout()
    plt.savefig('plots/figure_1_vert.pdf', format='pdf')
    plt.show()

fig, ax = plt.subplots(figsize=(10, 10))

# ax = plot_dataset(ax, data2)
ax = plot_decision_boundary(ax, X2, model2, alpha=0.75)

ax.plot(factual2[:, 0], factual2[:, 1], 'ko-', label='true trajectory', linewidth=2)
ax.set(aspect='equal')

for i, fact in enumerate(factual2):
    ax.text(fact[0] - 0.025, fact[1], r'$x_{}$'.format(i), va='center', ha='center',
            rotation='horizontal', fontsize=20, color='black', alpha=.7, zorder=1000)
ax.plot(cf11[j, :, 0], cf11[j, :, 1], 'k*', markersize=20, markeredgecolor='white',
        label='positive counterfactual')
ax.plot(cf22[j, :, 0], cf22[j, :, 1], 'w*', markersize=20, markeredgecolor='black',
        label='negative counterfactual')

for i in range(len(cf11[j])):
    shift = 0
    if i == 0:
        shift = 0.075
    ax.text(((factual2[i, 0] + factual2[i + 1, 0]) / 2) + 0.055, (factual2[i, 1] + factual2[i + 1, 1]) / 2 - (i-1) * - 0.02,
            r'$\bar S_{} = {}$'.format(i, np.round(scores2[i, 1], 4)), va='center', ha='center',
            rotation='horizontal', fontsize=20, color='black', alpha=.7, zorder=100)
    line1 = ax.annotate("", xy=(cf11[j, i, 0], cf11[j, i, 1]), xytext=(factual2[i, 0], factual2[i, 1]),
                        arrowprops=dict(arrowstyle="->", lw=3, color='blue', alpha=.7))
    line2 = ax.annotate("", xy=(cf22[j, i, 0], cf22[j, i, 1]), xytext=(factual2[i, 0], factual2[i, 1]),
                        arrowprops=dict(arrowstyle="->", lw=3, color='red', alpha=.7))

if j == 1:
    ax.legend(loc='upper left', fancybox=True, framealpha=0.2, prop={'size': 26}).set_zorder(1000)
ax.set_xlim([0.15, 0.7])
ax.set_ylim([0.15, 0.7])

ax.set(aspect='equal')
plt.tight_layout()
plt.savefig('plots/figure_2_square.png', format='png')
plt.show()

"""# generate cumulative average score
scores = np.random.rand(20)  # **** This replaces Ed's toy example scores, delete when running for case studies ****
cumulative_average_score = cumulative_average_trace(scores)
mean_trace = cumulative_average_score[-1]
print(f'Mean TraCE Score: {mean_trace:.2f}')

if plot:
    plt.plot(scores, label='Instantaneous TraCE')
    plt.plot(cumulative_average_score, label='Cumulative average TraCE')
    plt.xlabel('Step')
    plt.ylabel('TraCE score per step')
    plt.title(f'Average TraCE: {mean_trace:.2f}')
    plt.legend()
    plt.savefig('plots/trace_plot.pdf', format='pdf')
    plt.show()"""
