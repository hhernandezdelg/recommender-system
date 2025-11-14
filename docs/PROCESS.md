# Fashion Feature Learning Pipeline

## Introduction
This document explains the end to end workflow implemented in the fashion feature learning project. The goal is to guide any reviewer through the dataset preparation, the four assignment tasks, and the resulting analyses without requiring them to inspect the source code directly. The prose mirrors the structure of the course instructions so that each section can be mapped to a specific task notebook.

## Dataset Summary
The Fashion Product Images collection provides approximately forty four thousand pictures of clothing and accessories. Each image belongs to one of sixty nine article types. The assignment defines curated splits: the main classes train split with thirty nine classes, the main classes test split with the same label space, the new classes test split that covers thirty previously unseen classes, and dedicated support splits for both groups. All notebooks load the metadata stored in the `data` directory and access the raw images through the Kaggle mirror exposed by `kagglehub`. Every transform resizes an image to three channels of size sixty by eighty pixels and normalizes it with ImageNet statistics.

## Task 1: Supervised Classification
Task 1 focuses on training a convolutional classifier that predicts the article type for the main classes. The architecture named `CNET_HHD0_V44` contains five convolutional blocks followed by a dense head. Each block applies convolution, batch normalization, GELU activation, max pooling, and dropout. The classifier head maps the flattened representation to thirty nine logits and applies batch normalization before the softmax. Training uses cross entropy loss and the Adam optimizer with a learning rate of five times ten to the power of negative five. Performance is measured with accuracy and balanced accuracy to account for the imbalanced test split. The notebook `task1_classifier_training.ipynb` performs five fold cross validation, training on four folds and validating on the remaining fold. Metrics for every epoch are plotted and stored, and the final evaluation on the untouched main test set appears at the end of each fold.

Mathematically, let \(D = \{(x_i, y_i)\}_{i=1}^N\) with labels \(y_i \in \{1, \dots, C\}\) where \(C = 39\). The network outputs logits \(z_i \in \mathbb{R}^{C}\) which are turned into probabilities through softmax \(p_{i,c} = \frac{\exp(z_{i,c})}{\sum_{k=1}^{C} \exp(z_{i,k})}\). Cross entropy loss for a batch becomes
\[
\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^{B} \log p_{i, y_i}.
\]
Accuracy is computed as
\[
\text{accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{\hat{y}_i = y_i\},
\]
where \(\hat{y}_i = \arg\max_{c} p_{i,c}\), while balanced accuracy averages the per class recall
\[
\text{balanced accuracy} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{|D_c|} \sum_{(x_i,y_i) \in D_c} \mathbf{1}\{\hat{y}_i = c\},
\]
with \(D_c\) the subset of samples belonging to class \(c\). Reporting both measures satisfies the assignment requirement and highlights model behaviour on rare classes.

## Task 2: Metric Learning for Emerging Styles
Task 2 requires a model that learns dense embeddings able to generalize to new classes. The same convolutional backbone is reused but the classifier output is taken before the final softmax and normalized to unit length. Training employs the Triplet Loss function, which encourages embeddings of similar items to be closer together than embeddings of dissimilar items by at least a margin of zero point two. Because enumerating all triplets is infeasible, a balanced batch sampler creates batches containing an equal number of samples per class, and the triplet selector mines semi hard negatives whenever available while falling back to hard negatives otherwise. Training runs for seventy epochs with stochastic gradient descent using a learning rate of zero point one and momentum of zero point eight. Mixed precision speeds up computation and Recall at five on the validation batches monitors quality. After training, the notebook saves the embedding network weights to `notebooks/output/metric_learning/metric_model.pt`.

Let \(f_\theta(x) \in \mathbb{R}^{d}\) denote the embedding extracted by the network after \(L_2\) normalization. For every anchor \(x_a\) with positive \(x_p\) from the same class and negative \(x_n\) from a different class, the triplet loss is
\[
\mathcal{L}_{\text{triplet}} = \max\left( \|f_\theta(x_a) - f_\theta(x_p)\|_2^2 - \|f_\theta(x_a) - f_\theta(x_n)\|_2^2 + \alpha, 0 \right)
\]
with \(\alpha = 0.2\). Semi hard negatives satisfy \(\|f_\theta(x_a) - f_\theta(x_p)\|_2^2 < \|f_\theta(x_a) - f_\theta(x_n)\|_2^2 < \|f_\theta(x_a) - f_\theta(x_p)\|_2^2 + \alpha\), ensuring informative yet stable gradients. Model quality is evaluated with Recall at \(k\),
\[
\text{Recall@}k = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{\exists j \in \mathcal{N}_k(i) : y_j = y_i\},
\]
where \(\mathcal{N}_k(i)\) contains the \(k\) nearest neighbours of embedding \(f_\theta(x_i)\) under cosine similarity. Monitoring Recall at five ensures that at least one of the retrieved neighbours matches the query class.

## Task 3: Recommendation System
The recommendation task leverages the learned embeddings to retrieve similar products. For each scenario described in the assignment (main train to main test, main support to main test, new support to new test, and merged support to merged test), embeddings are extracted with the trained metric model. Cosine similarity is applied between normalized embeddings, and a k nearest neighbor search with k equal to three assigns labels by majority vote. Accuracy and balanced accuracy are reported for every scenario, matching the required table. The notebook also defines a confidence score for each recommendation, computed as the mean cosine similarity of the top three matches. By sweeping a threshold on that score, the system produces the error versus coverage curve requested in Task 3. The curve shows that targeting an error of ten percent yields roughly ninety five percent coverage, while one hundred percent coverage corresponds to an error of about eleven and a half percent.

Given a normalized test embedding \(t\) and support embeddings \(\{s_j\}\), cosine similarity simplifies to the dot product \(\text{sim}(t, s_j) = t^\top s_j\). For each test sample we select the set \(\mathcal{S}_k(t)\) of the \(k = 3\) most similar support embeddings and predict the label through majority vote:
\[
\hat{y}(t) = \text{mode}\{ y_j : s_j \in \mathcal{S}_k(t) \}.
\]
The confidence score used for the error coverage analysis equals the mean similarity of the retrieved neighbours
\[
c(t) = \frac{1}{k} \sum_{s_j \in \mathcal{S}_k(t)} \text{sim}(t, s_j).
\]
For a threshold \(\tau\), coverage and error rate are defined as
\[
\text{coverage}(\tau) = \frac{|\{t : c(t) \ge \tau\}|}{N}, \qquad \text{error}(\tau) = \frac{|\{t : c(t) \ge \tau, \hat{y}(t) \neq y(t)\}|}{|\{t : c(t) \ge \tau\}|}.
\]
Plotting error against coverage while varying \(\tau\) reproduces the required trade off curve.

## Task 4: Beyond Labels
Task 4 analyses the learned feature space qualitatively. First, t distributed stochastic neighbor embedding (t SNE) projects the embeddings of the first ten classes onto two dimensions for both the training and test splits, revealing distinct clusters for accessories and overlap for visually similar garments such as shirts and kurtas. The notebook repeats the t SNE analysis at the category level, which confirms that items from the same broader category stay close in the feature space. Next, a confusion matrix is computed using the ten class subset of the main support and main test splits, highlighting misclassifications predicted from the t SNE plots. The shirt versus t shirt study computes centroids for each class and visualizes the interpolation path between them, demonstrating that the feature manifold changes smoothly. Finally, the top three recommendation visualization shows how the confidence filtered retrieval behaves on real samples, aligning with the Task 3 recommendation system.

For each class \(c\) with embedding set \(\mathcal{E}_c\), the centroid is computed as \(m_c = \frac{1}{|\mathcal{E}_c|} \sum_{e \in \mathcal{E}_c} e\). The interpolation between shirts and t shirts samples points \(m(\lambda) = (1 - \lambda) m_{\text{shirt}} + \lambda m_{\text{tshirt}}\) for \(\lambda \in [0, 1]\), then retrieves the closest embeddings to each intermediate point. This demonstrates how the representation transitions smoothly between related garments. The confusion matrix uses the standard definition \(C_{ij} = |\{x : y(x) = i, \hat{y}(x) = j\}|\) but is normalised row wise to highlight percentages, as requested in the instructions. The qualitative interpretation of TSNE plots and centroid visualisations ties back to the questionnaire answers in the original submission.

## Implementation Highlights
The reusable pipeline lives inside `src/fashion_metric`, with dedicated modules for datasets, models, classifier training, metric learning, evaluation, and visualization. Every notebook imports from these modules, ensuring reproducibility and avoiding duplicated code. Regularization techniques include progressive dropout across the convolutional blocks, a dropout layer in the classifier head, and the use of color jitter plus small random rotations during training. Batch normalization and layer normalization stabilize optimization, while the dataloaders rely on multiple workers and pinned memory for efficient transfer to GPU. Metric learning uses mixed precision to reduce training time, and all scripts automatically detect whether CUDA is available.

## Results Summary
The classifier trained in Task 1 reaches balanced accuracy above the assignment requirement on the main test set, with cross validation demonstrating consistent performance. The metric learning model achieves Recall at five above ninety five percent on validation batches and enables a flexible recommendation system that delivers more than ninety five percent coverage when tolerating ten percent error. Visual analyses confirm that the learned embeddings meaningfully cluster similar styles and categories, demonstrating that the system supports Pradah Hinson’s vision for future fashion discovery.
