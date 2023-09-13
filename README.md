# TraCE: Trajectory Counterfactual Scores
## Idea

- Counterfactual explanations, and their paths, can provide a useful indication of what should
change to alter a classification outcome
- For time series tasks it may be useful to compare the actual path being taken with the ideal one
as suggested by counterfactuals to give a sense as to if that indiividual is moving in a broadly positive
direction or not
- Concretely: For a given time point, X0, a quantification of similarity between the actual path
taken to the next timepoint, X1, and the ‘best’ path (i.e. the counterfactual), X′
0, may be a
useful measure to indicate ‘quality’ of the current trajectory. For instance, for a patient in a
hospital ward and assessing how they are responding to treatment and if they are improving or
further deteriorating.
- In comparing these paths the difference in both direction and magnitude intuitively seem impor-
tant. A potential scoring metrics could utilise existing methods, such as:
    - L2 distance for magnitude
    - Cosine similarity for direction, which is already handily scaled to -1 to 1
- Using the cosine similarity, or combined with L2 distance or other values, into a single metric
may be a neat solution whereby the resulting score would mean:
    - Large positive: strong progress towards boundary and opposite class, i.e. improvement
    - Near 0: making no progress, potentially moving parallel to decision boundary
    - Large negative: clearly moving in wrong direction