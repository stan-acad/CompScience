class naiveBayes:
    def __init__(self):
        # dictionaries to hold the frequency of features and labels
        self.feature_counts = {}
        self.label_counts = {}
        self.total_samples = 0

    def train(self, X, y):
        """
        Train the Naive Bayes classifier.
        X: different weather conditions (outlook, temp, humidity, wind)
        y: is it advisable to play
        """
        self.total_samples = len(y)
        for features, label in zip(X, y):
            if label not in self.label_counts:
                self.label_counts[label] = 0
                self.feature_counts[label] = {}
            self.label_counts[label] += 1
            for feature in features:
                if feature not in self.feature_counts[label]:
                    self.feature_counts[label][feature] = 0
                self.feature_counts[label][feature] += 1

    def predict(self, X):
        """
        Predict the label for a given set of features.
        X: List of feature lists
        """
        predictions = []
        for features in X:
            label_probabilities = {}
            for label in self.label_counts:
                # calculate the prior probability of the label
                label_prob = self.label_counts[label] / self.total_samples
                # calculate the likelihood of the features given the label
                feature_prob = 1
                for feature in features:
                    if feature in self.feature_counts[label]:
                        feature_prob *= (self.feature_counts[label][feature] / self.label_counts[label])
                    else:
                        feature_prob *= 1e-6  # smoothing for unseen features
                # calculate the posterior probability
                label_probabilities[label] = label_prob * feature_prob
            # select the label with the highest posterior probability
            best_label = max(label_probabilities, key=label_probabilities.get)
            predictions.append(best_label)
        return predictions

if __name__ == "__main__":
    # sample data
    X_train = [['sunny', 'hot', 'high', 'weak'],
               ['sunny', 'hot', 'high', 'strong'],
               ['overcast', 'hot', 'high', 'weak'],
               ['rain', 'mild', 'high', 'weak'],
               ['rain', 'cool', 'normal', 'weak'],
               ['rain', 'cool', 'normal', 'strong'],
               ['overcast', 'cool', 'normal', 'strong'],
               ['sunny', 'mild', 'high', 'weak'],
               ['sunny', 'cool', 'normal', 'weak'],
               ['rain', 'mild', 'normal', 'weak'],
               ['sunny', 'mild', 'normal', 'strong'],
               ['overcast', 'mild', 'high', 'strong'],
               ['overcast', 'hot', 'normal', 'weak'],
               ['rain', 'mild', 'high', 'strong']]
    y_train = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

    # create and train the classifier
    classifier = naiveBayes()
    classifier.train(X_train, y_train)

    # test data
    X_test = [['sunny', 'mild', 'high', 'strong'],
              ['rain', 'mild', 'high', 'weak'],
              ['sunny', 'cool', 'normal', 'strong']]

    # predictions
    predictions = classifier.predict(X_test)
    print(predictions)  # Output: ['no', 'yes', 'yes']
