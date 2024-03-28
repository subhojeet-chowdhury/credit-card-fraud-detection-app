import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, precision_recall_curve, roc_curve, auc
from keras.models import load_model
import numpy as np
import time

def main():
    st.title("Credit Card Fraud Detection")

    # About section
    st.markdown("""
    ## About
    This app is designed to predict whether a credit card transaction is fraudulent or not.
    """)

    # Input section
    st.header("Test Data Input")

    # Option 1: Input single transaction data
    st.subheader("Enter Single Transaction Data")
    transaction_input = st.text_input("Enter transaction data (comma-separated):")

    # Option 2: Upload CSV file
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV file containing transaction data:", type=["csv"])

    # Prediction button
    if st.button("Start"):
        if transaction_input:
            # Process single transaction input and make prediction
            process_single_transaction(transaction_input)

        elif uploaded_file:
            # Process uploaded CSV file and make predictions
            process_uploaded_file(uploaded_file)

        else:
            st.error("Please enter transaction data or upload a CSV file")

def process_single_transaction(transaction_data):
    
    # Display the single transaction data in a beautiful table
    st.write("Received transaction data:")

    # Simulate data loading
    simulate_data_loading("Loading data...", 4)

    df_transaction = pd.DataFrame([transaction_data.split(",")], columns=["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "IsFraud"])
    st.table(df_transaction)

    # Display different visualizations
    st.write("Data loaded successfully for testing")

    simulate_data_loading("Presenting visualizations...", 3)

    #Input Data Visualizations
    st.subheader("Input Data Visualizations")
    st.write("Histogram of Transaction Amount")
    st.pyplot(plot_histogram(df_transaction))

    st.write("Bar plot of Class Distribution")
    st.pyplot(plot_class_distribution(df_transaction))

    #Start Predictions
    start_prediction(df_transaction)

def process_uploaded_file(uploaded_file):
    
    # Read CSV file
    df = pd.read_csv(uploaded_file)

    # Simulate data loading
    simulate_data_loading("Loading data...", 4)

    # Display the CSV file data in a beautiful table
    st.write("Received CSV file:")
    st.table(df)

    # Display different visualizations
    st.write("Data loaded successfully for testing")

    simulate_data_loading("Presenting visualizations...", 3)

    # Visualizations
    st.subheader("Input Data Visualizations")
    st.write("Histogram of Transaction Amount")
    st.pyplot(plot_histogram(df))

    st.write("Bar plot of Class Distribution")
    st.pyplot(plot_class_distribution(df))

    # start predictions
    start_prediction(df)


def simulate_data_loading(message, loadTime):
    # Simulate data loading with a sleep timer
    with st.spinner(message):
        time.sleep(loadTime)  # Simulate loading time (5 seconds)

def start_prediction(df):

    # Prediction button
    if st.button("Load model and start Prediction"):
       
        # Load the model using the cached function
        model = load_trained_model("oversampled_neural_network_model.keras")

        if model:
            st.write("Model loaded successfully!")

            ## Perform predictions and evaluation
            st.header("Perform Predictions and Evaluation")
            predict_and_evaluate(df, model)
        
        else:
            st.warning("Please upload a trained model.")



# Load trained model
@st.cache_resource
def load_trained_model(model_file):
    # Load the Keras model
    model = load_model(model_file)
    return model

# Perform predictions and evaluation
def predict_and_evaluate(df, model):
    if df is not None and model is not None:
        st.write("Data loaded successfully for testing")

        # Separate features and target variable
        X_test = df.drop(columns=["Class"])
        y_test = df["Class"]

        # Perform predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred.round())
        precision = precision_score(y_test, y_pred.round())
        recall = recall_score(y_test, y_pred.round())
        f1 = f1_score(y_test, y_pred.round())

        # Display evaluation metrics
        st.write("Evaluation Metrics:")
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)
        st.write("Prediction Time:", end_time - start_time, "seconds")

        # Generate insights and visualizations
        st.header("Insights and Visualizations")

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred.round())

        # Plot precision-recall curve
        plot_precision_recall_curve(y_test, y_pred)

        # Plot ROC curve
        plot_roc_curve(y_test, y_pred)

        # Plot feature importance
        plot_feature_importance(X_test, model)


def plot_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=20, kde=True)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.title("Histogram of Transaction Amount")
    return plt

def plot_class_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    return plt

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot()

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    st.pyplot()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot()

# Function to visualize feature importance
def plot_feature_importance(X_test, model):
    # Assuming model has attribute feature_importances_
    feature_importance = model.feature_importances_
    feature_names = X_test.columns
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    st.pyplot()


if __name__ == "__main__":
    main()
