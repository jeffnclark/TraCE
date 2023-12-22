# TraCE: Trajectory Counterfactual Scores

This repository hosts source code, and case study implementation, for [TraCE as introduced at NLDL 2024](https://openreview.net/pdf?id=Jgwi6CaTnU).

## TL;DR

Use TraCE scores to assess progress in high-dimensionality multi-step processes down to an easily interpretable single score by measuring alignment in true trajectory against theoretical trajectories towards counterfactuals:

<p style="text-align:center">
<img src="plots/fig1_paper_method.png" width="800">
</p>


## Abstract

Counterfactual explanations, and their associated algorithmic recourse, are typically leveraged to understand and explain predictions of individual instances coming from a black-box classifier. In this paper, we propose to extend the use of counterfactuals to evaluate progress in sequential decision making tasks. To this end, we introduce a model-agnostic modular framework, TraCE (Trajectory Counterfactual Explanation) scores, to distill and condense progress in highly complex scenarios into a single value. We demonstrate TraCE’s utility by showcasing its main properties in two case studies spanning healthcare and climate change.


## Requirments

Standard packages: Numpy, Pandas, Scikit-learn.

Optionally: Your choice of counterfactual example generator. In the demo notebook we use DiCE but the TraCE framework can be applied to counterfactuals from a variety of sources, as described below.

## Intended use
TraCE is a model-agnostic framework which can be applied across domains and interfaced with existing tools. Alongside a series of factual datapoints, users are required to provide counterfactual reference point(s) to calculate TraCE scores against. These reference points can take several forms, including:

* Model-generated counterfactual explanations using any existing counterfactual generation method
* Predetermined landmarks, such as those guided by experts
* Corpus of historical values
* Input from other scientific studies

Users can provide single or multiple sets of counterfactual reference point(s).

In our paper we demonstrate different possible implementations of TraCE:
1. Hospital case study, utilising TraCE scores to track progress for patients in an intensive care unit. In this example we provide TraCE with both desirable (successfully discharged) and undersirable (in-hospital mortality) counterfactuals, determined from a corpus of known outcome labels from the training set.
2. Sustainable development case study, utilising TraCE scores to evaluate countries' historical development against established pathways. In this example we provide TraCE with expert-derived pathways to track alignment against.

## Citation

```
@inproceedings{clark2023trace,
  title={TraCE: Trajectory Counterfactual Explanation Scores},
  author={Clark, Jeffrey Nicholas and Small, Edward Alexander and Keshtmand, Nawid and Wan, Michelle Wing Lam and Mayoral, Elena Fillola and Werner, Enrico and Bourdeaux, Christopher and Santos-Rodriguez, Raul},
  booktitle={Northern Lights Deep Learning Conference 2024},
  year={2023}
}
```
