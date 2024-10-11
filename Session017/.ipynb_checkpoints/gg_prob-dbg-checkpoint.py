import pandas as pd
import numpy as np
import pickle  # Using pickle for binary storage
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# File to store learned data (from user feedback) in binary format
learned_data_file = "gender_learnings_phonetic.pkl"
model_file = "gender_model.pkl"

# Load learned data from the pickle file
def load_learnings():
    if os.path.exists(learned_data_file):
        print(f"[DEBUG] Loading learnings from {learned_data_file}...")  # Step 1
        with open(learned_data_file, "rb") as file:
            learnings = pickle.load(file)
            print(f"[DEBUG] Loaded learnings: {learnings}")  # Step 2
            return learnings
    print(f"[DEBUG] No learned data found. Starting fresh.")  # Step 3
    return {}

# Save learned data to the pickle file
def save_learnings(learnings):
    print(f"[DEBUG] Saving learnings to {learned_data_file}...")  # Step 4
    with open(learned_data_file, "wb") as file:
        pickle.dump(learnings, file)
    print(f"[DEBUG] Learnings saved.")  # Step 5

# Load name-gender dataset from CSV
def load_dataset(filename):
    print(f"[DEBUG] Loading dataset from {filename}...")  # Step 6
    df = pd.read_csv(filename)
    print(f"[DEBUG] Dataset loaded with {len(df)} records.")  # Step 7
    return df[['name', 'gender', 'probability']]

# Train a gender prediction model using a dataset of names
def train_model(df):
    print("[DEBUG] Starting model training...")  # Step 8
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))  # Use character bigrams and trigrams
    print("[DEBUG] Vectorizing names...")  # Step 9
    X = vectorizer.fit_transform(df['name'])
    
    le = LabelEncoder()
    print("[DEBUG] Encoding gender labels...")  # Step 10
    y = le.fit_transform(df['gender'])

    # Get probability values as weights
    sample_weights = df['probability'].values
    print(f"[DEBUG] Sample weights: {sample_weights}")  # Step 11

    # Train-test split
    print("[DEBUG] Splitting data into training and testing sets...")  # Step 12
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

    print("[DEBUG] Training logistic regression model...")  # Step 13
    model = LogisticRegression()
    model.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate the model
    print("[DEBUG] Evaluating model performance...")  # Step 14
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[DEBUG] Model accuracy: {accuracy:.2f}")  # Step 15

    # Save the trained model
    save_model(model, vectorizer, le)

    return model, vectorizer, le

# Save the model to a pickle file
def save_model(model, vectorizer, le):
    print(f"[DEBUG] Saving trained model to {model_file}...")  # Step 16
    with open(model_file, "wb") as f:
        pickle.dump((model, vectorizer, le), f)
    print(f"[DEBUG] Model saved to {model_file}.")  # Step 17

# Load the model from the pickle file
def load_model():
    if os.path.exists(model_file):
        print(f"[DEBUG] Loading trained model from {model_file}...")  # Step 18
        with open(model_file, "rb") as f:
            model, vectorizer, le = pickle.load(f)
            print("[DEBUG] Model loaded successfully.")  # Step 19
            return model, vectorizer, le
    print(f"[DEBUG] No model found. Please train the model first.")  # Step 20
    return None, None, None

# Predict gender based on the name using the trained model
def predict_gender_ml(name, model, vectorizer, le, learnings):
    name_lower = name.lower()
    
    # Check if the name exists in the learned data first
    if name_lower in learnings:
        print(f"[DEBUG] Found '{name}' in learnings. Using learned gender: {learnings[name_lower]}")  # Step 21
        return learnings[name_lower]
    
    # If not found in learnings, use the ML model to predict
    print(f"[DEBUG] '{name}' not found in learnings. Using ML model for prediction...")  # Step 22
    name_features = vectorizer.transform([name])
    predicted_gender_idx = model.predict(name_features)[0]
    predicted_gender = le.inverse_transform([predicted_gender_idx])[0]
    
    print(f"[DEBUG] ML model prediction for '{name}': {predicted_gender}")  # Step 23
    return predicted_gender

# Main function to handle training and predictions
def main():
    parser = argparse.ArgumentParser(description="Train or predict gender based on name.")
    parser.add_argument('csvfile', nargs='?', help="Path to the CSV file containing name, gender, and probability columns (optional).")
    parser.add_argument('action', nargs='?', choices=['train', 'predict'], help="Action to perform: 'train' or 'predict'.")
    args = parser.parse_args()

    # Load learned data
    learnings = load_learnings()

    if args.csvfile and args.action == 'train':
        df = load_dataset(args.csvfile)
        model, vectorizer, le = train_model(df)
        print("[DEBUG] Model training completed. Ready for prediction.")
    else:
        model, vectorizer, le = load_model()
        if model is None:
            return

    # Continuous name prediction loop
    while True:
        name = input("Enter a name (or type 'exit' to quit): ").strip()
        if name.lower() == 'exit':
            break

        predicted_gender = predict_gender_ml(name, model, vectorizer, le, learnings)
        print(f"The predicted gender for '{name}' is: {predicted_gender}")

        correct = input(f"Is this correct? (yes/no): ").strip().lower()
        if correct == 'no':
            actual_gender = input("Please provide the correct gender (male/female): ").strip().lower()

            # Save this learning for future predictions
            learnings[name.lower()] = actual_gender
            save_learnings(learnings)
            print(f"[DEBUG] Learning saved: '{name}' is {actual_gender}.")

        continue_predicting = input("Do you want to enter another name? (yes/no): ").strip().lower()
        if continue_predicting == 'no':
            break

if __name__ == "__main__":
    main()
