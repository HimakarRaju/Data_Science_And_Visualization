import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

class GenderModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.is_trained = False
        self.raw_data = []
        
    def train(self, raw_data):
        """Train the model using raw data."""
        names, genders = zip(*raw_data)
        gender_map = {"M": 1, "F": 0}  # Map gender labels to numerical values
        y = np.array([gender_map[gender] for gender in genders])
        X = self.vectorizer.fit_transform(names)

        print("Class distribution before balancing:", Counter(y))

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = self.model.predict(X_test)
        
        # Print metrics
        print(f"Training Accuracy: {self.model.score(X_train, y_train) * 100:.2f}%")
        print(f"Testing Accuracy: {self.model.score(X_test, y_test) * 100:.2f}%")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        self.is_trained = True

    def learn(self, name, gender):
        """Update the model with a new example and append to raw data."""
        if not self.is_trained:
            print("Model needs to be trained first.")
            return

        gender_map = {"M": 1, "F": 0}
        X = self.vectorizer.transform([name])
        y = np.array([gender_map[gender]])

        # Train the model with the new data
        self.model.fit(X, y)
        append_to_raw_data([(name, gender)])
    
    def predict(self, name):
        """Predict gender based on the name."""
        if not self.is_trained:
            print("Model is not trained.")
            return None

        X = self.vectorizer.transform([name])
        prediction = self.model.predict(X)
        return prediction[0]  # Return 0 (Female) or 1 (Male)


def save_object(obj, filename):
    """Helper function to save any object."""
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def load_object(filename, default=None):
    """Helper function to load any object."""
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"{filename} not found. Returning default.")
        return default
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return default

def append_to_raw_data(new_data):
    """Append new data to the raw data file."""
    raw_data = load_object("train_raw_data.pkl", [])
    raw_data.extend(new_data)
    save_object(raw_data, "train_raw_data.pkl")

def get_user_choice():
    """Prompt user for an action choice."""
    while True:
        usr_ipt = input("Choose option \nA) Predict \nB) Train from file \nC) Train manually\nD) Model Info\nQ) Quit\n\n").lower()
        if usr_ipt in ["a", "b", "c", "d", "q"]:
            return usr_ipt
        print("Invalid choice. Please choose A, B, C, D, or Q.")

def train_model_from_csv():
    """Train the model using data from CSV."""
    try:
        df = pd.read_csv("name_gender.csv")  # Assuming CSV has 'name' and 'gender' columns
        raw_data = list(zip(df['name'], df['gender']))
        model = GenderModel()
        model.train(raw_data)
        save_object(model, "trained_model.pkl")  # Save trained model
        save_object(raw_data, "train_raw_data.pkl")  # Save raw data
    except Exception as e:
        print(f"Error reading CSV or training model: {e}")

def train_model_manually():
    """Train the model manually by entering names and genders."""
    train_raw_data = load_object("train_raw_data.pkl", [])
    new_data = []

    while True:
        name = input("Enter a name (or 'q' to quit): ").strip()
        if name.lower() == "q":
            break
        gender = input("Enter the gender (M/F): ").upper().strip()
        if gender in ["M", "F"]:
            new_data.append((name, gender))
        else:
            print("Invalid gender input. Please enter M or F.")

    if new_data:
        train_raw_data.extend(new_data)
        model = GenderModel()
        model.train(train_raw_data)  # Train with both old and new data
        save_object(train_raw_data, "train_raw_data.pkl")  # Save updated raw data
        save_object(model, "trained_model.pkl")  # Save the trained model

def main():
    """Main function to handle user interaction."""
    while True:
        model = load_object("trained_model.pkl", GenderModel())
        print("Model Loaded\n")
        user_choice = get_user_choice()
    
        if user_choice == "a":
            name = input("Enter a name: ").strip()
            predicted_gender = model.predict(name)
            if predicted_gender is not None:
                gender_str = "Male" if predicted_gender == 1 else "Female"
                print(f"Predicted gender: {gender_str}")
    
                user_feedback = input(f"Is the prediction correct? (Y/N): ").strip().upper()
                if user_feedback == "N":
                    correct_gender = input("Please enter the correct gender (M/F): ").strip().upper()
                    if correct_gender in ["M", "F"]:
                        model.learn(name, correct_gender)
                        save_object(model, "trained_model.pkl")
    
        elif user_choice == "b":
            train_model_from_csv()  # Train the model with CSV data
    
        elif user_choice == "c":
            train_model_manually()  # Manually add new data for training
    
        elif user_choice == "d":
            print("Model Info:")
            print(f"Trained: {'Yes' if model.is_trained else 'No'}")
            print(f"Model accuracy: {model.model.score(model.vectorizer.transform([x[0] for x in model.raw_data]), [1 if x[1] == 'M' else 0 for x in model.raw_data]) * 100:.2f}%")
    
        elif user_choice == "q":
            break

if __name__ == "__main__":
    main()
