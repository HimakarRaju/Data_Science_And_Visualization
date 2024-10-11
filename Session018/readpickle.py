import pickle
import os

# Open the pickle file in read binary mode
with open('gender_data.pkl', 'rb') as f:
    # Load the data from the file
    data = pickle.load(f)

# Now you can use the loaded data
print(data)
