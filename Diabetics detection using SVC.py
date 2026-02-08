import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_data = pd.read_csv('/content/diabetes.csv')

# Prepare data
def prepare_data(data):
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=19)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train and evaluate the model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_predicted = svm_classifier.predict(X_test)
    training_score = accuracy_score(svm_classifier.predict(X_train), y_train)
    test_score = accuracy_score(y_predicted, y_test)
    return training_score, test_score, svm_classifier

# Predict function
def predict_diabetes(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# Main function
def main():
    X_train, X_test, y_train, y_test, scaler = prepare_data(diabetes_data)
    training_score, test_score, trained_model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print("Training Accuracy:", training_score)
    print("Test Accuracy:", test_score)

    # Example predictions on a batch of input data
    example_data = pd.DataFrame([
        [5, 117, 92, 0, 0, 34.1, 0.337, 38],
        [10, 101, 76, 48, 180, 32.9, 0.171, 63],
        [3, 123, 100, 35, 240, 50.3, 0.666, 31]
    ], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    predictions = predict_diabetes(trained_model, scaler, example_data)
    
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            print(f"Sample {i + 1}: Diabetic")
        else:
            print(f"Sample {i + 1}: Non-diabetic")

    # Visualize prediction results
    plt.figure(figsize=(6, 4))
    plt.hist(predictions, bins=2, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['Non-diabetic', 'Diabetic'])
    plt.title('Prediction Distribution')
    plt.show()

if __name__ == "__main__":
    main()
