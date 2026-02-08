# Diabetes Prediction using Support Vector Machine (SVM)

## Overview

This project implements a **Diabetes Prediction System** using a **Support Vector Machine (SVM)** classifier.
The model is trained on a diabetes dataset to predict whether a person is diabetic or non-diabetic based on medical attributes such as glucose level, BMI, age, and other health indicators.

The workflow includes:

* Data loading and preprocessing
* Feature scaling using StandardScaler
* Training an SVM classification model
* Evaluating model performance using accuracy
* Making predictions for new patient samples
* Visualizing prediction distribution

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Dataset

The program expects a CSV file named:

```
diabetes.csv
```

The dataset should contain the following columns:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age
* Outcome (Target variable: 0 = Non-diabetic, 1 = Diabetic)

Place the dataset in the same directory as the script or update the file path in:

```python
pd.read_csv('/content/diabetes.csv')
```

---

## How the Project Works

### 1. Data Preparation

* The dataset is split into **training (80%)** and **testing (20%)** sets.
* Features are standardized using **StandardScaler** to improve model performance.

### 2. Model Training

* A **Linear SVM (SVC kernel='linear')** classifier is trained on the scaled training data.

### 3. Evaluation

* The model performance is evaluated using **training accuracy** and **test accuracy**.

### 4. Prediction

* The trained model predicts diabetes status for new input samples.

### 5. Visualization

* A histogram displays the distribution of predictions (Diabetic vs Non-diabetic).

---

## How to Run

1. Install required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

2. Place `diabetes.csv` in the project directory.

3. Run the script:

```bash
python diabetes_prediction.py
```

---

## Example Output

```
Training Accuracy: 0.78
Test Accuracy: 0.76
Sample 1: Non-diabetic
Sample 2: Diabetic
Sample 3: Diabetic
```

A histogram plot will also be displayed showing prediction distribution.

---

## Possible Improvements

* Hyperparameter tuning using GridSearchCV
* Trying other models (Random Forest, Logistic Regression)
* Deploying the model using Flask or Streamlit
* Adding real-time user input prediction interface

---

## Author

Developed as a Machine Learning mini-project for diabetes prediction using SVM.
