# Fashion Recommender Pipeline

## Introduction
This document explains the end to end workflow implemented in the fashion recommender learning project. The goal is to guide any reviewers and collaborators through the dataset preparation, the four main tasks, and the resulting analyses. The objective is that each section can be mapped to a specific task in the notebooks.

## Dataset Summary
The Fashion Product Images collection provides approximately 44k pictures of clothing and accessories. Each image belongs to one of sixty nine article types. The curated dataset splits are: the main classes train split with 39 classes, the main classes test split with the same label space, the new classes test split that covers 30 previously **unseen** classes, and dedicated support splits for both groups. All notebooks load the metadata stored in the `data` directory and access the raw images through the Kaggle mirror exposed by `kagglehub`. Every transform resizes an images and normalizes it.

## Task 1: Supervised Classification (Trasnfer learning)
Task 1 focuses on training a convolutional classifier that predicts the article type for the main classes. The architecture named `CNET_HHD0_V44` contains five convolutional blocks followed by a dense head. This convolutional network is used as transfer learning proxy, which has been used in different task domain (external domain tranfer learning). Each block applies convolution, batch normalization, GELU activation, max pooling, and dropout. The classifier head maps the flattened representation to 39 logits and applies batch normalization before the softmax. The training uses **cross entropy loss** and the **Adam optimizer**. Performance is measured with accuracy and balanced accuracy to account for the imbalanced test split. The notebook `task1_classifier_training.ipynb` performs five fold cross validation. Metrics for every epoch are plotted and stored, and the final evaluation on the untouched main test set appears at the end of each fold.

All mathematical expressions for cross-entropy, accuracy, and balanced accuracy are documented in `docs/metric_learning_math.ipynb`, which also includes derivations for the supervised training metrics.

## Task 2: Metric Learning for Emerging Styles
Task 2 requires a model that learns dense embeddings and is able to generalize to **new unseen classes**. The same convolutional backbone is reused but the classifier output (softmax function) is skipped and the logits are normalized to unit length. Training employs the Triplet Loss function, which encourages embeddings of similar items to be closer together than embeddings of dissimilar items by at least a defined **margin (0.2)**. The margin is a hyperparameter for this type of training, here we found 0.2 to be most optimal. Because enumerating all triplets is infeasible, a balanced batch sampler creates batches containing an equal number of samples per class, and the triplet selector mines semi-hard negatives whenever available while falling back to hard negatives otherwise. Training runs for seventy epochs with stochastic gradient descent and momentum. Mixed precision speeds up computation and Recall (@k=5) on the validation batches monitors quality. After training, the notebook saves the embedding network weights to `notebooks/output/metric_learning/metric_model.pt`.

For a theoretical background we follow the triplet-loss formulation introduced by [Schroff et al., FaceNet](https://arxiv.org/abs/1503.03832), which explains how semi-hard negative mining stabilises metric learning, and the explicit equations are summarised in `docs/metric_learning_math.ipynb`.

## Task 3: Recommendation System
The recommendation task leverages the learned embeddings to retrieve similar products. For each scenario: 

| Experiment                        | Train set       | Test set       |
|-----------------------------------|-----------------|----------------|
| Main train → main test            | main train      | main test      |
| Main support → main test          | main support    | main test      |
| New support → new test            | new support     | new test       |
| Merged support → merged test      | merged support  | merged test    |

the embeddings are extracted with the trained metric model. Cosine similarity is applied between normalized embeddings, and a k nearest neighbor search (k=3) assigns labels by majority vote. Accuracy and balanced accuracy are reported for every scenario, matching the required table. The notebook also defines a confidence score for each recommendation, computed as the mean cosine similarity of the top 3 matches. By sweeping a threshold on that score, the system produces an error-coverage curve. The curve shows that targeting an error of 10% yields roughly 95% coverage, while 100% coverage corresponds to an error of approximately 11.5%. The curve quantifies the trade off between coverage and error so that we can choose a confidence threshold that reflects application priorities. In some settings we accept a higher error rate to show more recommendations, while in others we restrict coverage to keep errors low. Bellow, two examples are drawn where characterizing such a curve can be crucial:

- E‑commerce recommendations: It is acceptable to tolerate a modest error rate to increase coverage because showing more relevant products usually outweighs a few irrelevant suggestions. Here we pick a lower confidence threshold to maximise how often we show recommendations.

- Medical decision support: It is critical to restrict errors even at the cost of hiding many suggestions because a wrong recommendation can mislead clinicians. Here we set a higher confidence threshold so only highly reliable suggestions are shown, even if coverage decreases.

The exact definitions of the mode-based prediction, confidence score, and the coverage/error trade-off are provided in `docs/metric_learning_math.ipynb`. Plotting error against coverage while varying $\tau$ reproduces the required trade off curve.

## Task 4: Beyond Labels
In this task, analyses of the learned feature space is taken place qualitatively. First, the distributed stochastic neighbor embedding (t-SNE) projects the embeddings of the first ten classes onto two dimensions for both the training and test splits, revealing distinct clusters for accessories and overlap for visually similar items such as shirts and kurtas. We repeat the same analysis (t-SNE) for the category level, which confirms that items from the same broader category stay close in the feature space. Next, a confusion matrix is computed using the ten class subset of the main support and main test splits, highlighting misclassifications predicted from the t SNE plots. The shirt versus t shirt study computes centroids for each class and visualizes the interpolation path between them, demonstrating that the feature manifold changes smoothly. Finally, the top three recommendation visualization shows how the confidence filtered retrieval behaves on real samples.

## Implementation Highlights
The reusable pipeline lives inside `src/fashion_metric`, with dedicated modules for datasets, models, classifier training, metric learning, evaluation, and visualization. Every notebook imports from these modules, ensuring reproducibility and avoiding duplicated code. Regularization techniques include progressive dropout across the convolutional blocks, a dropout layer in the classifier head, and the use of color jitter plus small random rotations during training. Batch normalization and layer normalization stabilize optimization, while the dataloaders rely on multiple workers and pinned memory for efficient transfer to GPU. Metric learning uses mixed precision to reduce training time, and all scripts automatically detect whether CUDA is available.
