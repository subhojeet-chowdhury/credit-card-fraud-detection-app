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
    if st.button("Start Prediction"):
        if transaction_input:
            # Process single transaction input and make prediction
            process_single_transaction(transaction_input)

        elif uploaded_file:
            # Process uploaded CSV file and make predictions
            process_uploaded_file(uploaded_file)

        else:
            st.error("Please enter transaction data or upload a CSV file")

def process_single_transaction(transaction_data):
    
    # with st.spinner("Loading data..."):

    # Convert transaction data to DataFrame
    values = transaction_data.split(",")
    values = [float(value) if index != len(values)-1 else int(value.strip('"')) for index, value in enumerate(values)]
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = pd.DataFrame([values], columns=columns)

    simulate_loading("Loading data...",3)
    st.success("Received transaction data:")
    st.table(df)

    st.success("Data loaded successfully for testing")

    with st.spinner("Presenting visualizations..."):
        #Input Data Visualizations
        st.header("Input Data Visualizations")
        st.subheader("Histogram of Transaction Amount")
        st.pyplot(plot_histogram(df))

        st.subheader("Bar plot of Class Distribution")
        st.pyplot(plot_class_distribution(df))

    # start prediction
    start_prediction(df, "single-transaction")

def process_uploaded_file(uploaded_file):
    
    with st.spinner("Loading data..."):
        df = pd.read_csv(uploaded_file)

    simulate_loading("Loading data...",3)
    # Display the CSV file data in a beautiful table
    st.success("Received CSV file:")
    st.table(df)
    # Display different visualizations
    st.success("Data loaded successfully for testing")
    
    with st.spinner("Presenting visualizations..."):
        # Visualizations
        st.header("Input Data Visualizations")
        st.subheader("Histogram of Transaction Amount")
        st.pyplot(plot_histogram(df))

        st.subheader("Bar plot of Class Distribution")
        st.pyplot(plot_class_distribution(df))
    
    # start prediction
    start_prediction(df, "uploaded-file")


def start_prediction(df, input_type):

    # Loading saved model from directory
    simulate_loading("Loading saved model...",3)
    model_name = "oversampled_neural_network_model.keras"
    model = load_trained_model(model_name)
    st.success("Model loaded successfully")
    st.write(model_name)

    simulate_loading("Performing predictions...",3)
    st.header("Predictions and Evaluations")

    predict_and_evaluate(df,model,input_type);
    st.balloons()

# Load trained model
@st.cache_resource
def load_trained_model(model_file):
    model = load_model(model_file)
    return model

# Perform predictions and evaluation for uploaded file
def predict_and_evaluate(df, model, input_file):
    if df is not None and model is not None:

        # Separate features and target variable
        X_test = df.drop(columns=["Class", "Time"])
        y_test = df["Class"]

        # Perform predictions
        start_time = time.time()
        y_pred_proba = model.predict(X_test)
        end_time = time.time()
        
        # Convert probability predictions to binary classification results
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Determine whether the prediction is true or false for each transaction
        prediction_result = ["True Positive" if pred == 1 and actual == 1 else
                             "False Positive" if pred == 1 and actual == 0 else
                             "True Negative" if pred == 0 and actual == 0 else
                             "False Negative" for pred, actual in zip(y_pred, y_test)]

        # Add the prediction result to the DataFrame for visualization
        df["Prediction Result"] = prediction_result

        # Display evaluation metrics
        st.subheader("Evaluation Metrics:")
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)
        st.write("Prediction Time:", end_time - start_time, "seconds")

        # Display prediction result for each transaction
        st.subheader("Prediction Results for Each Transaction")
        st.write(df[["Class", "Prediction Result"]])


        if input_file == "uploaded-file":
            # Generate insights and visualizations
            st.header("Insights and Visualizations")
            # Plot confusion matrix
            st.pyplot(plot_confusion_matrix(y_test, y_pred))
            # Plot precision-recall curve
            st.pyplot(plot_precision_recall_curve(y_test, y_pred))
            # Plot ROC curve
            st.pyplot(plot_roc_curve(y_test, y_pred))


def simulate_loading(message, loadTime):
    # Simulate data loading with a sleep timer
    with st.spinner(message):
        time.sleep(loadTime)

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
    return plt

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    return plt

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
    return plt

if __name__ == "__main__":
    main()
