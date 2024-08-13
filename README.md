# Real-Estate-Price-Prediction
Overview
This project is a simple web application built using Python, Streamlit, and Scikit-learn, designed to predict real estate prices based on a set of features. The model employed is a linear regression model trained on a dataset of real estate properties.

Key Features
Interactive User Interface: The app allows users to input various features like the distance to the nearest MRT station, the number of convenience stores, latitude, and longitude.
Real-time Prediction: Upon inputting the features, the app predicts the house price of a unit area in real-time.
Model Training: A linear regression model is trained on a dataset to make predictions.
Dependencies
Ensure the following Python libraries are installed in your environment:

pandas: For data manipulation and analysis.
streamlit: For creating the web application interface.
scikit-learn: For building and training the machine learning model.
IPython: For displaying embedded Vimeo videos (if needed).
You can install these dependencies using pip:

bash
Copy code
pip install pandas streamlit scikit-learn IPython
Dataset
The dataset used in this project contains information on real estate properties, including features such as:

Distance to the nearest MRT station: How far the property is from the closest Mass Rapid Transit (MRT) station.
Number of convenience stores: The number of convenience stores near the property.
Latitude and Longitude: The geographical coordinates of the property.
House price of unit area: The target variable representing the price per unit area of the house.
How It Works
Load the Dataset: The dataset is loaded using pandas from a CSV file located on your local machine.
Feature Selection: The model uses four key features to predict house prices.
Train-Test Split: The data is split into training and testing sets to evaluate the model's performance.
Model Training: A linear regression model is trained on the training data.
Streamlit Interface: The Streamlit app takes user inputs for the four features and predicts the house price using the trained model.
How to Run the Application
Ensure all dependencies are installed.
Run the Streamlit app using the command:
bash
Copy code
streamlit run app.py
The app will open in a web browser where you can input the required features and get the predicted house price.
Code Structure
Data Loading:

python
Copy code
real_estate_data = pd.read_csv("C:/Users/HP/Downloads/Real_Estate_data.csv")
Loads the real estate dataset from the specified path.

Feature and Target Selection:

python
Copy code
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'
Selects the features and target variable for model training.

Data Splitting:

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Splits the data into training and testing sets.

Model Training:

python
Copy code
model.fit(X_train, y_train)
Trains the linear regression model on the training data.

Streamlit Interface:
The app takes user inputs, processes them, and makes a prediction. The prediction is displayed on the app's interface.

Troubleshooting
Missing Libraries: If you encounter errors related to missing libraries, ensure all dependencies are correctly installed.
File Path: Make sure the dataset's file path is correct and accessible.
Input Validation: Ensure all input fields in the app are filled to avoid prediction errors.
Future Enhancements
Additional Features: Incorporating more features for more accurate predictions.
Model Improvement: Testing and implementing more sophisticated models like Random Forests or Gradient Boosting.
User Interface: Enhancing the Streamlit interface for a better user experience.
