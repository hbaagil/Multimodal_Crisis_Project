from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


def train_logistic_regression(X_train, y_train):
    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Create and train the LogReg
    logreg_classifier = LogisticRegression(max_iter=1000)
    logreg_classifier.fit(X_train, y_train)

    print(f"✅ Model train on logistic regression binary classification")

    return logreg_classifier

def predict_binary_logistic_regression(logreg_classifier, X):
    # Predict binary labels (0 or 1) using the trained LogReg classifier



    # Make predictions using the model
    predictions = logreg_classifier.predict(X)

    # Map predictions to labels
    # 0 is mapped to "informative" and 1 is mapped to "not informative"
    labels = ["informative" if prediction == 0 else "not informative" for prediction in predictions]

    print(f"✅ Model predict on logistic regression binary classification")

    return labels
