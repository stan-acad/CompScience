def calcDistance(point1, point2):
    """Compute the Euclidean distance between two points."""
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def get_neighbors(training_data, test_point, k):
    """Find the k nearest neighbors of the test_point from the training_data."""
    distances = [(train_point, calcDistance(test_point, train_point[:-1])) for train_point in training_data]
    distances.sort(key=lambda x: x[1])
    return [distances[i][0] for i in range(k)]

def predict_classification(training_data, test_point, k):
    """Predict the class of a test point based on majority voting of k nearest neighbors."""
    neighbors = get_neighbors(training_data, test_point, k)
    classes = [neighbor[-1] for neighbor in neighbors]
    return max(set(classes), key=classes.count)

# example usage
if __name__ == "__main__":
    # data (features + class label)
    training_data = [
        [2.5, 2.1, 'A'], [1.3, 3.4, 'A'], [3.6, 1.8, 'B'],
        [4.2, 2.9, 'B'], [2.0, 3.0, 'A'], [3.8, 3.2, 'B']
    ]

    test_point = [3.0, 2.5]
    
    k = 3
    predicted_class = predict_classification(training_data, test_point, k)
    
    print("Predicted Class:", predicted_class)