---

# Naive Bayes Classifier

This repository contains implementations of Naive Bayes classifiers for sentiment analysis tasks using two different datasets: Titanic dataset and Airline Tweets dataset. The purpose of this project is to demonstrate the application of Naive Bayes algorithms for classification tasks and to analyze sentiment in different contexts.

## Table of Contents
1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Description](#description)
4. [Results](#results)
5. [Usage](#usage)

## Introduction

Naive Bayes classifiers are a family of probabilistic classifiers based on Bayes' theorem with the assumption of independence between features. They are commonly used for text classification tasks, including sentiment analysis. In this project, we implement and compare Naive Bayes classifiers using two distinct datasets: the Titanic dataset and an Airline Tweets dataset.

## Datasets

### Titanic Dataset
The Titanic dataset contains information about passengers aboard the Titanic, including their demographics and whether they survived the disaster. This dataset is used for binary classification to predict passenger survival based on features such as gender, age, fare, and passenger class.

### Features:
1. **PassengerId**: Unique identifier for each passenger.
2. **Survived**: Whether the passenger survived or not (0 = No, 1 = Yes).
3. **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
4. **Name**: Passenger's name.
5. **Sex**: Passenger's gender.
6. **Age**: Passenger's age.
7. **SibSp**: Number of siblings/spouses aboard.
8. **Parch**: Number of parents/children aboard.
9. **Ticket**: Ticket number.
10. **Fare**: Passenger's fare.
11. **Cabin**: Cabin number.
12. **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

### Target Class:
- **Survived**: Whether the passenger survived or not.
  - 0 = No
  - 1 = Yes

### Insights:
- **Demographic Analysis**: Analysis involves exploring the demographics of passengers, such as age, gender, and class, to understand their distribution and impact on survival rates.
- **Feature Importance**: Identifying the most important features for predicting passenger survival and understanding their significance in determining survival outcomes.
- **Model Evaluation**: Performance of classification models, such as Gaussian Naive Bayes, is evaluated using metrics like accuracy, precision, recall, and F1-score to assess their effectiveness in predicting passenger survival.

### Airline Tweets Dataset

The Airline Tweets dataset consists of tweets related to various airlines, along with sentiment labels indicating whether the tweet expresses positive, neutral, or negative sentiment towards the airline. This dataset is used for multi-class classification to predict sentiment based on the text of the tweets and other features such as confidence scores and airline.

### Features:
1. **Airline**: Airline associated with the tweet.
2. **Text**: Text content of the tweet.
3. **Airline_sentiment**: Sentiment label of the tweet (positive, neutral, negative).
4. **Airline_sentiment_confidence**: Confidence score for the sentiment label.

### Target Class:
- **Airline_sentiment**: Sentiment label of the tweet.
  - Positive
  - Neutral
  - Negative

### Insights:
- **Sentiment Analysis**: Analysis involves analyzing the sentiment distribution across different airlines to understand customer perceptions and satisfaction levels.
- **Text Preprocessing**: Preprocessing of text data, including tokenization, removal of stop words, and lemmatization, to prepare the text for classification.
- **Model Training and Evaluation**: Performance of classifiers like Multinomial Naive Bayes and hybrid Naive Bayes is evaluated using accuracy, confusion matrix, and classification report to assess their ability to predict sentiment from tweets.

## Description

### Titanic Dataset Code
- The Titanic dataset code preprocesses the dataset by removing irrelevant columns and handling missing values.
- It visualizes the distribution of passenger survival, gender, passenger class, and fare.
- It trains a Gaussian Naive Bayes classifier to predict passenger survival based on features like gender, age, fare, and passenger class.
- It evaluates the classifier's performance using accuracy, confusion matrix, and classification report.

### Airline Tweets Dataset Code
- The Airline Tweets dataset code preprocesses the dataset by removing missing values and performing text preprocessing.
- It visualizes the distribution of sentiment labels and confidence scores across different airlines.
- It trains multiple Naive Bayes classifiers (Gaussian, Multinomial) using different feature sets, including text features extracted using TF-IDF.
- It evaluates the classifiers' performance using accuracy, confusion matrix, and classification report.
- It implements a hybrid Naive Bayes classifier by combining probabilities from Gaussian and Multinomial classifiers with different feature sets.

## Results

### Titanic Dataset
- The Gaussian Naive Bayes classifier achieves a certain accuracy in predicting passenger survival.
- The classifier's performance is evaluated using a confusion matrix and classification report, showing precision, recall, and F1-score for each class.

### Airline Tweets Dataset
- Multiple Naive Bayes classifiers are trained and evaluated for sentiment analysis on airline tweets.
- The classifiers achieve varying accuracies and performance metrics for predicting sentiment.
- A hybrid Naive Bayes classifier combining Gaussian and Multinomial classifiers with different feature sets is implemented and evaluated for improved performance.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/<username>/naive-bayes-classifier.git
   ```
2. Navigate to the repository directory:
   ```
   cd naive-bayes-classifier
   ```
