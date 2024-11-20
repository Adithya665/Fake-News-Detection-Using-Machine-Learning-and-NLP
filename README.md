# Fake-News-Detection-Using-Machine-Learning-and-NLP
This repository contains the implementation code for the dissertation "Identifying Fake News Using Real-Time Analytics", focusing on detecting fake news articles using machine learning and natural language processing (NLP) techniques.

## Repository Structure
lstm_model.ipynb: Jupyter Notebook implementing the Long Short-Term Memory (LSTM) model for fake news classification.
svm_rf_lr_models.ipynb: Jupyter Notebook implementing Support Vector Machine (SVM), Random Forest (RF), and Logistic Regression (LR) models for fake news classification.

# Project Overview
Fake news detection is critical in combating misinformation in the digital age. This project leverages advanced NLP and machine learning techniques to classify news articles as fake or real. The developed system aims to provide high accuracy and real-time classification for mitigating the spread of fake news.

## Key Features
Data Preprocessing: Includes data cleaning, stemming, tokenization, and TF-IDF vectorization.
Machine Learning Models: Combines traditional and deep learning approaches:
LSTM for sequential pattern detection.
SVM, RF, and LR for robust classification.
Evaluation Metrics: Models assessed using accuracy, precision, recall, and F1-score.

# Installation and Setup
Prerequisites
Python 3.8 or higher
Jupyter Notebook
Required libraries: pandas, numpy, scikit-learn, tensorflow, matplotlib, seaborn, nltk, imblearn

# Data Preprocessing and Models
Cleaning: Removing HTML tags, special characters, and stopwords.
Normalization and Stemming: Ensures consistent data representation.
Feature Extraction: TF-IDF vectorization for numerical representation of text.
Balancing: Handled class imbalance using SMOTE.

# Models
LSTM: Sequential deep learning model to capture context.
SVM: Effective for high-dimensional classification.
Random Forest: Ensemble model leveraging decision trees.
Logistic Regression: Baseline model for interpretable results.


# Results

Model	
Logistic Regression	98.98	
Random Forest	99.71	
Support Vector Machine	99.52
LSTM	99.45	

# Dataset
link - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection/data.

