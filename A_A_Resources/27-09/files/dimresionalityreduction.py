'''
Start
  |
  v
Input: Dataset, Variance Threshold
  |
  v
For Each Feature:
    Compute Mean of Feature
    |
    Compute Variance of Feature
    |
    Is Variance > Threshold?
       /   \
     Yes    No
      |      |
  Keep Feature  Remove Feature
  |
  v
Output: Reduced Dataset with Selected Features
  |
  v
End



'''


def variance_threshold(data, threshold=0.1):
    selected_features = []
    
    # Transpose data to iterate over features instead of samples
    for feature in zip(*data):
        mean = sum(feature) / len(feature)
        variance = sum((x - mean) ** 2 for x in feature) / len(feature)
        
        # Select feature if variance exceeds threshold
        if variance > threshold:
            selected_features.append(feature)
    
    # Transpose back to original shape
    return list(zip(*selected_features))

# Example usage
data = [[1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1]]
reduced_data = variance_threshold(data)
print(f"Reduced data: {reduced_data}")
