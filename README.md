# Fashion Recommender System

This repository contains a complete implementation of a fashion-feature learning pipeline that combines supervised transfer learning and metric learning. The system addresses two goals: (1) train a convolutional neural network (CNN) via transfer learning to classify article types in the “main classes” split, and (2) learn robust embeddings with a triplet-loss metric-learning approach that supports recommendation systems and qualitative analyses. The project was originally developed as part of the 2AMM10 course (Assignment 1) and has since then been reorganised into a clean, modular code base with detailed documentation.

The main objectives, full mathematical derivations, experiment descriptions, and final findings are documented in `docs/PROCESS.md`. This README keeps a high-level overview so visitors can quickly grasp the repo structure, install the environment, and locate key components.

---

## 1. Project Overview

### Goals and Highlights
1. **Transfer learning**: The custom CNN `CNET_HHD0_V44` is trained with cross-entropy loss on the fashion dataset’s “main” classes. It uses a series of convolutional blocks with dropout to deliver accurate predictions (Task 1).
2. **Metric learning**: The same backbone is reused with a triplet loss and hard/semi-hard negative mining to learn embeddings that generalise to unseen classes (Task 2).
3. **Recommendation system**: Using the learned embeddings, the project builds a k-nearest-neighbour recommender that provides “similar items” suggestions with confidence scores (Task 3).
4. **Qualitative analysis**: Visualisations (t-SNE, confusion matrices, centroid transitions) explain how the embeddings capture class relationships and category hierarchies (Task 4).

The repository’s structure supports reproducibility, with all shared logic living in `src/fashion_metric`, per-task notebooks in `notebooks/`, and consolidated documentation in `docs/`.

---

## 2. Environment Installation

The project is tested with Python 3.12+. It can run via [uv](https://github.com/astral-sh/uv) (recommended) or via a traditional virtual environment.

### Using uv (recommended)

```bash
uv python install 3.12              # ensures the Python 3.12 toolchain is available
uv python pin 3.12                  # optional: pin the project to Python 3.12
uv sync --python 3.12               # create .venv and install dependencies
uv run python -m ipykernel install --user \
    --name fashion-metric-learning-py312 \
    --display-name "fashion-metric-learning (Python 3.12)"
uv run jupyter lab                  # or: uv run jupyter notebook
```

### Using classic virtualenv (if uv is unavailable)

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab                        # or: jupyter notebook
```

### Data and Paths

1. Place CSV metadata under `data/` (as provided in the assignment).
2. The notebooks expect images from the Kaggle “Fashion Product Images” dataset. When run, they will download via `kagglehub`.
3. Checkpoints are stored under:
   - `notebooks/output/` for Task 1 (per-fold metrics and model weights)
   - `notebooks/output/metric_learning/` for Task 2 (embedding network weights)
   - `notebooks/output/` for Task 3 and Task 4 figures

---

## 3. Repository Structure

### `src/fashion_metric/` (code base)
- `data.py`: dataset utilities (`FashionDataset`, transforms, dataloader builders).
- `models.py`: CNN architecture `CNET_HHD0_V44` with feature extractor and classifier head.
- `classifier_training.py`: `TransferLearnP2` training class, plus trainer function for Task 1.
- `metric_learning.py`: triplet-loss trainer, batch sampler, and recall computation.
- `evaluation.py`: embedding extraction, k-NN inference, coverage/error utilities.
- `visualization.py`: t-SNE plotting, confusion matrices, centroid transitions, top-k recommendations.
- `__init__.py`: convenience exports for direct use in notebooks.

### `notebooks/`
- `task1_classifier_training.ipynb`: trains the transfer-learning CNN with 5-fold cross-validation (Task 1).
- `task2_metric_learning.ipynb`: trains the triplet-loss embedding model (Task 2).
- `task3_knn_evaluation.ipynb`: evaluates four k-NN scenarios, builds error-vs-coverage plots (Task 3).
- `task4_visual_analysis.ipynb`: reproduces every qualitative analysis (t-SNE, confusion matrix, shirt↔T-shirt interpolation, recommendations) (Task 4).

Each notebook begins with an “Assignment Context” cell so you can immediately see how it maps to each task.

### `docs/`
- `PROCESS.md`: the long-form report covering the dataset, mathematical formulations, training details, results, and key insights. This is the best starting point if you want to understand the project without running code.
- `metric_learning_math.ipynb`: collects the metric learning equations and intermediate derivations.

## 4. Running the Experiments

1. **Task 1 (Classifier)**: Open `notebooks/task1_classifier_training.ipynb`, set `ENABLE_TRAINING = True`, and run all cells. This trains the CNN on 5 folds and reports accuracy and balanced accuracy.
2. **Task 2 (Metric Learning)**: Open `notebooks/task2_metric_learning.ipynb`, set `ENABLE_TRAINING = True`, and run all cells. This trains the triplet-loss model and saves the embedding weights.
3. **Task 3 (Recommendations)**: With the Task 2 weights saved, open `notebooks/task3_knn_evaluation.ipynb` and run it. It logs the k-NN metrics for all four scenarios and produces the error-vs-coverage figure.
4. **Task 4 (Visual Analyses)**: Open `notebooks/task4_visual_analysis.ipynb` to compute t-SNE projections, confusion matrices, shirt↔T-shirt interpolations, and the top-3 recommendation visualisations based on the learned embeddings.

All outputs are saved in the directories described above. The `docs/PROCESS.md` file explains the interpretation of each experiment, including any references to the assignment text and questionnaire answers.

---

## 5. Read `docs/PROCESS.md`

It covers:

1. **Dataset**: how the splits are organised and why the resolution/normalisation choices were made.
2. **Task 1**: architecture explanation, cross-entropy objective, accuracy and balanced accuracy metrics.
3. **Task 2**: triplet-loss formulation, semi-hard mining rationale, recall@k definition, and training design.
4. **Task 3**: confidence scoring scheme, error vs coverage curve derivation, and scenario comparisons.
5. **Task 4**: interpretation of t-SNE clusters, centroid transitions, and confusion matrix patterns.

The README therefore stays concise and defers all narrative detail to the documentation.
