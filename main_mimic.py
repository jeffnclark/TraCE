import data.datasets as datasets
from helpers.plotters import *
from helpers.funcs import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import classification_report


# import mimic dataset
data1 = datasets.generate_expired('mimic_scaled')  # [2000:3500]

print(data1.columns)
mimic_features = data1.drop(['y'], axis=1).columns

x1 = data1.iloc[:, 1:3]  # ensure data is 2d
y1 = data1.y

print(y1.value_counts())

# create model for scenario 1
model1 = MLPClassifier(hidden_layer_sizes=(
    10, 10), random_state=1, max_iter=700).fit(x1, y1)  # get target
# print(f'Balanced accuracy score:', balanced_accuracy_score(model1.predict(x1), y1))
print(classification_report(y1, model1.predict(x1)))
X1 = data1[mimic_features[1:3]]


# factual at t=0, t=1, t=2
factual = np.array([[0.5, 0.1],
                    [0.55, 0.25],
                    [0.45, 0.4]])

cf1 = np.array([[[0.4, 0.6],
                 [0.475, 0.65]],
                [[0.4, 0.6],
                 [0.475, 0.65]]
                ])
cf2 = np.array([[[0.545, 0.555],
                 [0.575, 0.55]],
                [[0.625, 0.3],
                 [0.615, 0.4]]
                ])

test_x0 = np.array([0, 0])
test_x1 = np.array([1, 1])
test_xprime = np.array([1, 0])
test_xstar = np.array([-1, 0])

test = score(test_x0, test_x1, test_xprime, test_xstar)
# scores
scores = np.zeros((2, 2))
for i in range(len(factual) - 1):
    xt = factual[i, :]
    xt1 = factual[i + 1, :]
    for j in range(cf1.shape[2]):
        x_prime = cf1[j, i, :]
        x_star = cf2[j, i, :]
        scores[i, j] = score(xt, xt1, x_prime, x_star)

# plotting
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5), sharey=True)

ax[0] = plot_dataset(ax[0], data1)
ax[0] = plot_decision_boundary(ax[0], X1, model1)

ax[1] = plot_dataset(ax[1], data1)
ax[1] = plot_decision_boundary(ax[1], X1, model1)


for j, ax in enumerate(ax):
    ax.plot(factual[:, 0], factual[:, 1], 'ko-', label='true trajectory')
    for i, fact in enumerate(factual):
        ax.text(fact[0] + 0.02, fact[1], r'$x_{}$'.format(i), va='center', ha='center',
                rotation='horizontal', fontsize=16, color='black', alpha=.7, zorder=100)
    ax.plot(cf1[j, :, 0], cf1[j, :, 1], 'k*', markersize=12,
            markeredgecolor='white', label='positive counterfactual')
    ax.plot(cf2[j, :, 0], cf2[j, :, 1], 'w*', markersize=12,
            markeredgecolor='black', label='negative counterfactual')

    for i in range(len(cf1[j])):
        ax.text(((factual[i, 0] + factual[i + 1, 0]) / 2) + 0.05, (factual[i, 1] + factual[i + 1, 1]) / 2,
                r'$R_{} = {}$'.format(i, np.round(scores[i, j], 2)), va='center', ha='center',
                rotation='horizontal', fontsize=16, color='black', alpha=.7, zorder=100)
        line1 = ax.annotate("", xy=(cf1[j, i, 0], cf1[j, i, 1]), xytext=(factual[i, 0], factual[i, 1]),
                            arrowprops=dict(arrowstyle="->", lw=1, color='green', alpha=.7))
        line2 = ax.annotate("", xy=(cf2[j, i, 0], cf2[j, i, 1]), xytext=(factual[i, 0], factual[i, 1]),
                            arrowprops=dict(arrowstyle="->", lw=1, color='red', alpha=.7))

    if j == 0:
        ax.legend(loc='lower right', fancybox=True,
                  framealpha=0.2, prop={'size': 12})
    ax.set_xlim([0.35, 0.7])
    ax.set_ylim([0, 0.7])

plt.savefig('plots/mimic_figure_1.pdf', format='pdf')
plt.show()
print('stop')
