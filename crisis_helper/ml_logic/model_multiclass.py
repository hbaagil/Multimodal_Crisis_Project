from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


def train_logistic_regression_multi(X_train, y_train):
    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Create and train the LogReg
    logreg_classifier_multi = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
    logreg_classifier_multi.fit(X_train, y_train)

    print(f"✅ Model trained on logistic regression multiclass classification")

    return logreg_classifier_multi

def predict_multiclass_logistic_regression(logreg_classifier, X):
    # Predict binary labels (0 or 1) using the trained LogReg classifier
    predictions = logreg_classifier.predict(X)

    # Define the label mapping
    label_mapping = {
        0: "affected_individuals",
        1: "infrastructure_and_utility_damage",
        2: "injured_or_dead_people",
        3: "missing_or_found_people",
        4: "not_humanitarian",
        5: "other_relevant_information",
        6: "rescue_volunteering_or_donation_effort",
        7: "vehicle_damage"
    }

    # Map predictions to labels using the label mapping
    labels = [label_mapping[prediction] for prediction in predictions]

    print(f"✅ Model predict on logistic regression binary classification")

    return labels
