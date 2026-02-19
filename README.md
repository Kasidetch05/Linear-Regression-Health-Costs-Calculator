# Linear-Regression-Health-Costs-Calculator

---
Introduction

This project uses a Deep Neural Network (DNN) built with TensorFlow and Keras to predict individual healthcare expenses. This is a regression problem, meaning the model's goal is to output a continuous numerical value (the predicted cost) rather than a category (like "spam" or "ham").

---

The Goal

The primary objective of this model is to estimate annual individual medical expenses with high accuracy.
Success Criterion: The model must achieve a Mean Absolute Error (MAE) of less than $3,500. This means, on average, the model's prediction should be within $3,500 of the actual cost.

---

The Data Used
To predict the cost, the model analyzes several key features about an individual:

- Age

- Body Mass Index (BMI)

- Number of Children

- Sex

- Smoker Status (Yes/No - a very strong predictor!)

- Geographic Region

Data Preparation Steps
Before the neural network can use the data, I prepared it:

1. Categorical Conversion: Non-numerical details like sex (male/female) and smoker status (yes/no) were converted into simple numerical flags (0 or 1).

2. Region Encoding: The region feature was processed using One-Hot Encoding. This means I created a separate 0/1 column for each possible region (e.g., region_northwest, region_southeast).

3. Train/Test Split: The entire dataset was split into two parts: 80% for training the model and 20% for testing its performance on data it has never seen before.

---

The Prediction Engine (Model Architecture)
The prediction is handled by a simple, three-layer Sequential Neural Network.

1. Input Layer: This layer receives the prepared features (age, BMI, smoker status, etc.) for a single person.
2. Hidden Layers (Two): The model uses two deep layers, each with 64 units and the ReLU activation function. These layers work to identify complex, non-linear relationships between the input features (e.g., how the combination of being a smoker and having a high BMI affects costs).
3. Output Layer: This final layer has a single output unit with no activation function (a linear output). It simply outputs the predicted dollar amount for the healthcare expense.

Training: The model is trained using the Adam optimizer and the Mean Absolute Error (MAE) as its loss function, which guides the network to minimize the average prediction error.

---

Performance Snapshot
The model was rigorously tested on unseen data to check its real-world effectiveness.
- Test MAE Result: The final model's performance on the test set is calculated.
- MAE Interpretation: The MAE tells us the average difference between the predicted cost and the actual cost. For example, an MAE of $3,200 means, on average, the prediction is off by that amount.

The script includes visual plots that compare the predicted expenses against the actual expenses. A perfect model would show all data points lying exactly on a 45-degree diagonal line. Our final plot shows how close the predictions are to this ideal line.

---

Code Overview
The script is logically divided into cells that manage the entire machine learning pipeline:
- CELL 1-4: Handles all Data Loading and Preprocessing, including feature conversion and the train/test split.

- CELL 5: Builds the Neural Network architecture (Dense layers).

- CELL 6: Trains the model and plots the training history (Loss and MAE over epochs).

- CELL 7-10: Evaluates Model Performance on the test set, calculates the final MAE score, and visualizes the prediction errors.

- CELL 11-12: Includes an Alternative Model (a deeper network) and the Final Test Cell to ensure the best-performing model is selected for the final results.

---


