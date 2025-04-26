# Spam-Ham-GrowthLink_Internship-Project

Overview
This project aims to classify SMS messages as either Spam or Ham (non-spam) using Logistic Regression and TF-IDF Vectorization for text processing. The dataset used for training the model is spam.csv, which contains labeled SMS messages.

Files Included
spam.csv: Dataset containing labeled SMS messages (Ham or Spam).

Spam_Ham_Classification.ipynb: Jupyter notebook containing the entire process for data preprocessing, model training, and evaluation.

Requirements
To run this project, you will need the following libraries installed in your Python environment:

numpy

pandas

seaborn

matplotlib

scikit-learn

You can install these libraries using the following command:

bash
Copy
Edit
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
Importing Libraries: We import necessary libraries such as pandas, numpy, matplotlib, and sklearn for data processing and machine learning tasks.

Loading and Preprocessing Data: The dataset is loaded from the spam.csv file and preprocessed:

Missing values are handled.

Labels (spam and ham) are converted into binary values: 0 for Spam and 1 for Ham.

Text Vectorization: We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical features.

Model Training: A Logistic Regression model is trained on the training data.

Model Evaluation: The model is evaluated using:

Accuracy

Confusion Matrix (visualized using Seaborn heatmap)

Classification Report (precision, recall, F1-score)

Prediction: The model is tested with sample messages to predict whether they are Spam or Ham.

Example Predictions
Here are a few example predictions made by the model:

Input Message: "Well, I'm gonna finish my bath now. Have a good...fine night."

Predicted: Ham

Input Message: "WINNER!! As a valued network customer you have been selected to receive a $900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."

Predicted: Spam
