# Stock Price Prediction using Machine Learning in Python

A machine learning project that predicts whether buying a particular stock (Tesla) will be profitable or not, using historical OHLC (Open, High, Low, Close) price data.

---

## 📌 Project Overview

This project uses Tesla stock price data from **January 2010 to December 2017** to train and evaluate multiple ML classification models. The goal is to predict a binary signal — whether the next day's closing price will be higher than the current day's (i.e., whether to buy or not).

---

## 📂 Repository Structure

```
├── stock_price_prediction.ipynb   # Main Jupyter Notebook
├── Tesla.csv                      # Dataset (Tesla stock prices 2010–2017)
└── README.md                      # Project documentation
```

---

## 📊 Dataset

- **Source:** Tesla Stock Price Historical Data
- **Period:** January 1, 2010 – December 31, 2017
- **Rows:** 1,692
- **Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Adj Close`

> Stock market is closed on weekends and holidays, so some dates are missing — this is expected.

---

## 🔧 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost` | XGBoost classifier |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook stock_price_prediction.ipynb
```

### 4. Update the dataset path
In the notebook, change the CSV path to match your local setup:
```python
df = pd.read_csv(r"C:\Users\YourName\Downloads\Tesla.csv")  # Windows
# or
df = pd.read_csv("Tesla.csv")  # if placed in the same folder
```

---

## 🧠 ML Models Trained

- **Logistic Regression**
- **Support Vector Machine (SVM)** — polynomial kernel
- **XGBoost Classifier**

Models are evaluated using the **ROC-AUC score** on both training and validation sets (90/10 split).

---

## 📈 Results

| Model | Training AUC | Validation AUC |
|---|---|---|
| Logistic Regression | 0.519 | 0.544 |
| SVM (poly kernel) | 0.472 | 0.447 |
| XGBoost | 0.964 | 0.573 |

> XGBoost shows signs of overfitting. Logistic Regression generalizes better on this dataset.

---

## 🔍 Key Steps

1. **EDA** — Visualizing closing price trend, distribution plots, and boxplots
2. **Feature Engineering** — Extracting day/month/year, creating `is_quarter_end`, `open-close`, and `low-high` features
3. **Normalization** — StandardScaler applied before model training
4. **Model Evaluation** — ROC-AUC score and Confusion Matrix

---

## 📝 Conclusion

The accuracy achieved by all models is close to 50%, which is no better than random guessing. This highlights the inherent difficulty of stock market prediction. Possible improvements include using more data, additional features (sentiment analysis, technical indicators), or deep learning models like LSTM.

---

## 🙏 Acknowledgements

- Project based on the tutorial from [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/stock-price-prediction-using-machine-learning-in-python/)
- Dataset: Tesla historical stock price data
