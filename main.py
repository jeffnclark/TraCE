import numpy as np

import data.datasets as datasets
from helpers.plotters import *
from helpers.funcs import *
from sklearn.neural_network import MLPClassifier

# settings
samples = 300
seed = 10
noise = 0.2
plot = True

# define data generator
generator = datasets.create_blobs  # get data generator

# create data set and model for scenario 1
data1 = generator(samples, seed, noise, centre=(0.75, 0.75))  # generate data
x1 = data1.iloc[:, :2]  # ensure data is 2d
y1 = data1.y

model1 = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1, max_iter=700).fit(x1, y1)  # get target
X1 = data1[["x1", "x2"]]

# create data set and model for scenario 2
data2 = generator(samples, seed, noise, centre=(1, 0.6))  # generate data
x2 = data2.iloc[:, :2]  # ensure data is 2d
y2 = data2.y

model2 = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1, max_iter=700).fit(x2, y2)  # get target
X2 = data2[["x1", "x2"]]

# factual at t=0, t=1, t=2
factual = np.array([[0.5, 0.1],
                    [0.52, 0.25],
                    [0.45, 0.4]])

cf1 = np.array([[[0.425, 0.5],
                 [0.5, 0.54]],
                [[0.425, 0.5],
                 [0.5, 0.54]]
                ])
cf2 = np.array([[[0.545, 0.555],
                 [0.575, 0.55]],
                [[0.625, 0.45],
                 [0.58, 0.5]]
                ])

test_x0 = np.array([0, 0])
test_x1 = np.array([0, 1])
test_xprime = np.array([[1, 0], [-1, 1]])
test_func = lambda a, b, c : (np.linalg.norm(a-c) - np.linalg.norm(b-c)) / np.sqrt(2)
test = score(test_x0, test_x1, test_xprime[0], func=test_func)
# scores
scores = np.zeros((2, 2))
for i in range(len(factual) - 1):
    xt = factual[i, :]
    xt1 = factual[i + 1, :]
    for j in range(cf1.shape[2]):
        x_prime = cf1[j, i, :]
        x_star = cf2[j, i, :]
        temp1 = score(xt, xt1, x_prime)
        temp2 = score(xt, xt1, x_star)
        scores[i, j] = 1/2*(temp1 - temp2)

        #scores[i, j] = score(xt, xt1, [x_prime], [x_star], method='avg')

# plotting
if plot:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5), sharey=True)

    ax[0] = plot_dataset(ax[0], data1)
    ax[0] = plot_decision_boundary(ax[0], X1, model1)

    ax[1] = plot_dataset(ax[1], data2)
    ax[1] = plot_decision_boundary(ax[1], X2, model2)

    for j, ax in enumerate(ax):
        ax.plot(factual[:, 0], factual[:, 1], 'ko-', label='true trajectory')
        for i, fact in enumerate(factual):
            ax.text(fact[0] + 0.02, fact[1], r'$x_{}$'.format(i), va='center', ha='center',
                    rotation='horizontal', fontsize=16, color='black', alpha=.7, zorder=100)
        ax.plot(cf1[j, :, 0], cf1[j, :, 1], 'k*', markersize=12, markeredgecolor='white', label='positive counterfactual')
        ax.plot(cf2[j, :, 0], cf2[j, :, 1], 'w*', markersize=12, markeredgecolor='black', label='negative counterfactual')

        for i in range(len(cf1[j])):
            ax.text(((factual[i, 0] + factual[i + 1, 0]) / 2) + 0.05, (factual[i, 1] + factual[i + 1, 1]) / 2,
                    r'$R_{} = {}$'.format(i, np.round(scores[i, j], 4)), va='center', ha='center',
                    rotation='horizontal', fontsize=16, color='black', alpha=.7, zorder=100)
            line1 = ax.annotate("", xy=(cf1[j, i, 0], cf1[j, i, 1]), xytext=(factual[i, 0], factual[i, 1]),
                                arrowprops=dict(arrowstyle="->", lw=1, color='green', alpha=.7))
            line2 = ax.annotate("", xy=(cf2[j, i, 0], cf2[j, i, 1]), xytext=(factual[i, 0], factual[i, 1]),
                                arrowprops=dict(arrowstyle="->", lw=1, color='red', alpha=.7))

        if j == 0:
            ax.legend(loc='lower right', fancybox=True, framealpha=0.2, prop={'size': 12})
        ax.set_xlim([0.35, 0.7])
        ax.set_ylim([0, 0.7])

    plt.savefig('plots/figure_1.pdf', format='pdf')
    plt.show()


# generate cumulative average score
scores = np.random.rand(20) # **** This replaces Ed's toy example scores, delete when running for case studies ****
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
    plt.show()