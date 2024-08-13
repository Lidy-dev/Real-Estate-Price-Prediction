## Overview
This project is a simple web application built using Python, Streamlit, and Scikit-learn, designed to predict real estate prices based on a set of features. The model employed is a linear regression model trained on a dataset of real estate properties.

## Key Features
- **Interactive User Interface**: The app allows users to input various features like the distance to the nearest MRT station, the number of convenience stores, latitude, and longitude.
- **Real-time Prediction**: Upon inputting the features, the app predicts the house price of a unit area in real-time.
- **Model Training**: A linear regression model is trained on a dataset to make predictions.

## Dependencies
Ensure the following Python libraries are installed in your environment:

- `pandas`: For data manipulation and analysis.
- `streamlit`: For creating the web application interface.
- `scikit-learn`: For building and training the machine learning model.
- `IPython`: For displaying embedded Vimeo videos (if needed).

You can install these dependencies using pip:

```bash
pip install pandas streamlit scikit-learn IPython
```

## Dataset
The dataset used in this project contains information on real estate properties, including features such as:

- **Distance to the nearest MRT station**: How far the property is from the closest Mass Rapid Transit (MRT) station.
- **Number of convenience stores**: The number of convenience stores near the property.
- **Latitude and Longitude**: The geographical coordinates of the property.
- **House price of unit area**: The target variable representing the price per unit area of the house.

## How It Works

### Data Loading:
The dataset is loaded using `pandas` from a CSV file located on your local machine.

```python
real_estate_data = pd.read_csv("C:/Users/HP/Downloads/Real_Estate_data.csv")
```

### Feature and Target Selection:
The model uses four key features to predict house prices.

```python
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'
```

### Train-Test Split:
The data is split into training and testing sets to evaluate the model's performance.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training:
A linear regression model is trained on the training data.

```python
model.fit(X_train, y_train)
```

### Streamlit Interface:
The Streamlit app takes user inputs for the four features and predicts the house price using the trained model.

## How to Run the Application
1. Ensure all dependencies are installed.
2. Run the Streamlit app using the command:

```bash
streamlit run app.py
```

The app will open in a web browser where you can input the required features and get the predicted house price.


## Visualizations

### Correlation Heatmap
The correlation heatmap shows the relationships between the features and the target variable.
![Correlation Heatmap](https://github.com/Lidy-dev/Real-Estate-Price-Prediction/blob/main/Corr_RealEstate.png)

### Feature Distributions
The distributions of the features are plotted to observe their spread and behavior.

![Feature Distributions](https://github.com/Lidy-dev/Real-Estate-Price-Prediction/blob/main/Features_RealEstate.png)

### Actual vs. Predicted Prices
This plot compares the actual house prices with the prices predicted by the model.

![Actual vs. Predicted Prices](https://github.com/Lidy-dev/Real-Estate-Price-Prediction/blob/main/ActualVSPredicted.png)

### Residual Plot
The residual plot shows the difference between actual prices and predicted prices.

![Residual Plot](https://github.com/Lidy-dev/Real-Estate-Price-Prediction/blob/main/Residual_RealEstate.png)


