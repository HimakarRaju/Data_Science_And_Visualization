def load_raw_data():
    
    """Load the raw training data."""
    
    try:
        with open("train_raw_data.pkl", "rb") as f:
            raw_data = pickle.load(f)
            # Check if the data is a DataFrame, convert to list of tuples
            if isinstance(raw_data, pd.DataFrame):
                raw_data = raw_data.to_records(index=False).tolist()
            return raw_data
    except FileNotFoundError:
        print("Training data not found. Initializing empty raw data.")
        return []  # Return an empty list if no raw data exists
    except Exception as e:
        print("Error loading training data:", str(e))
    return []

def train_Model():
    data = load_raw_data()


def main():
    while True:
        user_choice = ("""
        Please choose an option
        A) Train \n
        B) Predict \n
        C) Model Info \n
        D) Quit \n
        """).upper()
    
        if user_choice == "A":
            train_Model()
        elif user_choice == "B":
            predict()
        elif user_choice == "C":
            model_info()
        elif user_choice == "D":
            break
    return print("Process Finished")

        