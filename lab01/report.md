# Lab 01 — Machine Learning Systems

## 1. Objective

The main objective of this laboratory was to set up a Python environment and evaluate two machine learning algorithms using the Breast Cancer Wisconsin dataset.

The following models were selected for testing:

- Logistic Regression
- Random Forest

During the experiment, I evaluated several performance indicators, including model accuracy, training duration, inference speed, model file size, and memory consumption. The obtained results were then compared against the resource limitations of four deployment environments: Cloud, Edge, Mobile, and TinyML.

## 2. Experimental Methodology

To begin, I configured a virtual environment using Python 3.11.8 and installed all necessary libraries.

The Breast Cancer Wisconsin dataset was imported with the `load_breast_cancer()` function. It was split into training and testing subsets in a 70/30 proportion. I applied stratified sampling and used `random_state=42` to ensure that the experiment could be reproduced.

The following configurations were used:

- Logistic Regression with `max_iter=1000`
- Random Forest consisting of 100 trees

A random seed of 42 was applied throughout the experiment.

Initially, both algorithms were trained and their classification accuracy was evaluated using the test subset.

For performance benchmarking, I performed one warm-up execution followed by five training runs. The median training duration was selected as the representative measurement.

Inference performance was tested using a single test sample. After one preliminary prediction to warm up the model, I executed 100 predictions and calculated their median execution time.

Finally, Joblib was used to save the trained models and determine their storage requirements. Peak memory consumption was also recorded separately for the training and inference stages.

## 3. Baseline Performance

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.9415 |
| Random Forest | 0.9357 |

According to the experimental results, Logistic Regression demonstrated slightly higher classification accuracy compared to Random Forest. It achieved a score of 0.9415, whereas Random Forest reached 0.9357.

## 4. System Resource Consumption

| Model | Training Time (s) | Inference Time (ms) | Model Size (KB) | Peak Training Memory (MiB) | Peak Inference Memory (MiB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4388 | 0.0573 | 1.03 | 231.1 | 230.6 |
| Random Forest | 0.5254 | 6.1039 | 284.07 | 232 | — |
