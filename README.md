# Spam-Ham Classification Project

## Overview

This project aims to classify SMS messages as either **Spam** or **Ham** (non-spam) using **Logistic Regression** and **TF-IDF Vectorization** for text processing. The dataset used for training the model is `spam.csv`, which contains labeled SMS messages.

## Files Included

- **spam.csv**: Dataset containing labeled SMS messages (Ham or Spam).
- **Spam_Ham_Classification.ipynb**: Jupyter notebook containing the entire process for data preprocessing, model training, and evaluation.

## Requirements

To run this project, you will need the following libraries installed in your Python environment:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install these libraries using the following command:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
How to Run the Jupyter Notebook
Clone or Download the Repository:

Clone the repository using:

bash
Copy
Edit
git clone <repository_url>
Or download the project as a zip file and extract it.

Navigate to the Project Directory:

bash
Copy
Edit
cd <project_directory>
Launch Jupyter Notebook: To start the Jupyter Notebook, use the following command in the terminal:

bash
Copy
Edit
jupyter notebook
Open the Notebook: After launching Jupyter, open the Spam_Ham_Classification.ipynb file in your web browser.

Run the Notebook Cells:

Click on each cell and press Shift + Enter to execute the code.

The notebook will guide you through the steps of loading the dataset, preprocessing, training the model, and evaluating the performance.

Steps Followed in the Notebook
Importing Libraries:
Necessary libraries such as pandas, numpy, matplotlib, and sklearn are imported for data processing and machine learning tasks.

Loading and Preprocessing Data:
The dataset is loaded from the spam.csv file and preprocessed:

Missing values are handled.

Labels (spam and ham) are converted into binary values: 0 for Spam and 1 for Ham.

Text Vectorization:
TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the text data into numerical features suitable for machine learning models.

Model Training:
A Logistic Regression model is trained on the vectorized data to classify SMS messages as Spam or Ham.

Model Evaluation:
The model is evaluated using:

Accuracy

Confusion Matrix (visualized using Seaborn heatmap)

Classification Report (precision, recall, F1-score)

Prediction:
The model is tested with sample messages to predict whether they are Spam or Ham.

Example Predictions
Here are a few example predictions made by the model:

Input Message: "Well, I'm gonna finish my bath now. Have a good...fine night."

Predicted: Ham

Input Message: "WINNER!! As a valued network customer you have been selected to receive a $900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."

Predicted: Spam

Project Documentation
SMS Spam-Ham Classification using Logistic Regression
Introduction
This project focuses on classifying SMS messages as Spam or Ham (Not Spam) using traditional machine learning techniques. It uses the spam.csv dataset and applies logistic regression after transforming the text data using TF-IDF vectorization. The main objective is to develop an efficient model that can distinguish between spam and legitimate messages with high accuracy.

Dataset Overview
Source: spam.csv

Columns:

v1: Label (ham or spam)

v2: The actual SMS message text

Data Preprocessing
Text and Label Extraction:

X = v2 (Text Messages)

Y = v1 (Labels: Spam or Ham)

Label Encoding:

Spam = 0

Ham = 1

Data Splitting:

80% training data

20% testing data

Vectorization:
TF-IDF (TfidfVectorizer) is used with the following parameters:

min_df=1 (terms must appear in at least one document)

stop_words='english' (removes common English stop words)

lowercase=True (converts text to lowercase)

Model Training
Algorithm used: Logistic Regression

Training code:

python
Copy
Edit
model = LogisticRegression()
model.fit(X_train_features, Y_train)
Evaluation
Accuracy on Training Set:
Computed using accuracy_score.

Accuracy on Test Set:
Computed similarly to training set.

Confusion Matrix:
Visualized using Seaborn heatmap.

Classification Report:
Includes precision, recall, and F1-score.

python
Copy
Edit
print(classification_report(Y_test, prediction_on_test_data, target_names=['Ham', 'Spam']))
Results
The model performs well on both training and test data, demonstrating generalization capability. The confusion matrix and metrics indicate that the classifier can accurately detect spam messages.

Predictive System
A simple predictive pipeline is included to check custom SMS messages:

python
Copy
Edit
input_your_mail = ["Your message here"]
features = feature_extraction.transform(input_your_mail)
prediction = model.predict(features)
Depending on the output (1 = Ham, 0 = Spam), the message type is printed.

Conclusion
This machine learning pipeline effectively classifies SMS messages with high accuracy using Logistic Regression and TF-IDF. It can be extended with more advanced models or incorporated into real-world spam detection systems.

typescript
Copy
Edit

Now you can paste this into your `README.md` file! Let me know if you need any further changes or addition.
