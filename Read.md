# Census Income Classification and Segmentation Project

## Overview

This project implements a machine learning pipeline to:

1. Predict whether an individual earns more than $50,000 annually using XGBoost classification.
2. Perform segmentation of individuals using KMeans clustering.

The pipeline is fully modular and built using Object-Oriented Programming (OOP). All modules are imported and executed through a single entry point (main.py).

All outputs including predictions, evaluation metrics, fairness metrics, segmentation results, and plots are automatically saved in the outputs/ directory.

---

## Dataset

Dataset: U.S. Census Bureau Income Dataset

Records: 199,523 individuals  
Features: Demographic and employment attributes  
Target variable: income_gt_50k (binary classification)

---

## Pipeline Architecture

The pipeline performs the following steps:

1. Load raw dataset using LoadDataset class
2. Perform preprocessing:
   - Missing value imputation
   - Categorical encoding for modeling
   - Binary target creation
3. Perform Exploratory Data Analysis (EDA)
4. Perform segmentation using KMeans clustering
5. Train XGBoost classifier using train/validation/test split
6. Evaluate performance using:
   - F1 score
   - Balanced accuracy
   - ROC-AUC
   - PR-AUC
7. Evaluate fairness using:
   - Statistical Parity Difference (SPD)
   - Equal Opportunity Difference (EOD)
8. Save all outputs and plots automatically

---

## Object-Oriented Design

The pipeline uses modular OOP components:

LoadAndProcessDataset → data loading and preprocessing  
XGBPipeline → model training and prediction  
Evaluator → performance and fairness evaluation  
SegmentationPipeline → clustering and segmentation  
EDAPlotter → exploratory data analysis  

All components are executed from main.py.

---

## Project Structure

project/

├── README.md  
├── requirements.txt  

├── dataset/  
│   ├── census-bureau.data  
│   └── census-bureau.columns  

├── src/  
│   ├── main.py  
│   ├── LoadAndProcessDataset.py  
│   ├── xgb_model.py  
│   ├── evaluator.py  
│   ├── metrics.py  
│   ├── segmentation.py  
│   └── eda_plots.py  

├── outputs/  

---

## Requirements

Python 3.9+

Install dependencies:

pip install -r requirements.txt

---

## How to Run

Navigate to src:

cd src

Run pipeline:

python main.py

---

## Outputs

Generated automatically in outputs/:

xgb_feature_importance.csv  
xgb_metrics.csv  
fairness_summary.csv  
segmentation_with_clusters.csv  
cluster_profile.csv  
EDA plots  
Cluster visualization plots (cluster_plot and cluster_sizes)  
Elbow plot to determine number of clusters for K-Means
Feature Importance plots
---

## Reproducibility

Random seed fixed at:

random_state = 42

Ensures consistent results.