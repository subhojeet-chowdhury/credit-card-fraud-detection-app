# 📦 Credit Card Fraud Detection App

## Description

This project implements various machine learning models to detect instances of credit card fraud. It includes preprocessing steps such as data cleaning and feature engineering, as well as model training and evaluation.

## Hosted App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detectionn.streamlit.app/)

## About

This Streamlit app is designed to help users detect potential instances of credit card fraud. It provides an interface for users to input transaction data, either manually or by uploading a CSV file, and predicts whether each transaction is fraudulent or not.

## Model Performance

| **Model**                    | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **AUC**   |
|-----------------------------|:------------:|:-------------:|:----------:|:------------:|:---------:|
| RandomForest                |   0.999853   |    0.073171   |  0.926829  |   0.987013   | 0.955975  |
| DecisionTree                |   0.999754   |    0.079268   |  0.920732  |   0.935950   | 0.928279  |
| PlainNeuralNetwork          |   0.999484   |    0.180894   |  0.819106  |   0.874187   | 0.845750  |
| WeightedNeuralNetwork       |   0.991801   |    0.075203   |  0.924797  |   0.165274   | 0.280431  |
| UnderSampledNeuralNetwork   |   0.998142   |    0.000703   |  0.999297  |   0.996997   | 0.998146  |
| OverSampledNeuralNetwork    |   0.997353   |    0.004065   |  0.995935  |   0.394525   | 0.565167  |

## Features

- Input single transaction data manually or upload a CSV file
- Visualize data distribution and model predictions
- Evaluate model performance with metrics such as accuracy, precision, recall, and F1 score

## Getting Started

To get started with the app, simply click the "Hosted App" badge above to launch it in your browser. You can then interact with the app by providing transaction data and exploring the visualizations and prediction results.


To clone the project and make your custom changes, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/subhojeet-chowdhury/credit-card-fraud-detection.git
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Make your changes and add advanced features

4. Run the main script to preprocess the data, train the models, and evaluate their performance:

    ```
    streamlit run streamlit_app.py
    ```

## Further Reading

For further information and resources related to credit card fraud detection and machine learning, consider exploring the following:

- [Understanding Credit Card Fraud Detection](https://www.experian.com/blogs/ask-experian/credit-education/preventing-fraud/what-is-credit-card-fraud/)
- [Machine Learning for Credit Card Fraud Detection](https://towardsdatascience.com/credit-card-fraud-detection-a1c7e1b75f59)
- [Streamlit Documentation](https://docs.streamlit.io/)
