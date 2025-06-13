# Bitcoin Price Prediction using Machine Learning

This project demonstrates how to predict Bitcoin prices using various machine learning algorithms. The model is trained on historical price data and attempts to forecast future values based on trends and patterns.

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)

## 🚀 Project Overview

The goal of this project is to apply machine learning techniques to predict Bitcoin closing prices. The project explores data preprocessing, feature engineering, and prediction using regression models like:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LSTM (for future integration)

## 📊 Dataset

The dataset used for this project is historical Bitcoin price data, which includes:

- Date
- Open, High, Low, Close prices
- Volume

You can download the dataset from [Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data) or any reliable cryptocurrency API.

## 🛠 Technologies Used

- Python
- Scikit-learn
- Pandas
- Numpy
- Matplotlib / Seaborn
- Jupyter Notebook
- XGBoost
- (Optional) TensorFlow/Keras for LSTM

## 💻 Installation

```bash
git clone https://github.com/yourusername/Bitcoin-price-prediction-using-machine-learning.git
cd Bitcoin-price-prediction-using-machine-learning
pip install -r requirements.txt
````

## ▶️ Usage

1. Open the notebook:

```bash
jupyter notebook model.ipynb
```

2. Run the cells sequentially to:

   * Load and preprocess the data
   * Visualize trends
   * Train models
   * Evaluate performance
   * Predict future prices

## 📈 Model Training & Evaluation

* Data was split into training and testing sets.
* Metrics used:

  * Mean Squared Error (MSE)
  * R² Score
* Feature selection was done based on correlation analysis.

## ✅ Results

* The Random Forest model provided the most accurate predictions on test data with the lowest RMSE.
* Linear Regression showed moderate accuracy.
* XGBoost improved generalization and reduced overfitting.

## 🔮 Future Work

* Integrate LSTM for sequence prediction
* Deploy model using Flask/FastAPI
* Real-time prediction using live APIs (e.g., CoinAPI)
* Incorporate sentiment analysis from Twitter/news sources

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.


The sample dataset is also available in the repository as ```bitcoin.csv```
