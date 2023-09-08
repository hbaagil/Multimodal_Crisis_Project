from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


def train_logreg_multiclass(X_train, y_train):
    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Create and train the LogReg for multi-class classification
    logreg_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
    logreg_classifier.fit(X_train, y_train)

    print(f"✅ Model train on logistic regression multiclass classification")

    return logreg_classifier

def predict_logreg_multiclass(logreg_classifier, X):
    # Predict binary labels (0 or 1) using the trained LogReg classifier

    print(f"✅ Model predict on logistic regression multiclass classification")

    return logreg_classifier.predict(X)
