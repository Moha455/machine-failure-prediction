ALLA MED
# ⚙️ Machine Failure Prediction

A machine learning project for predicting industrial machine failures using operational sensor data.

The project covers the complete workflow from exploratory data analysis to model deployment with Streamlit.

## 🎯 Objective

The goal is to predict whether a machine is likely to fail based on operating conditions such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

This is an imbalanced binary classification problem:

- `0` → Normal operation
- `1` → Machine failure

## 🔎 Exploratory Data Analysis

The EDA included:

- Class distribution analysis
- Histograms and KDE plots
- Boxplots
- Correlation analysis
- Feature relationships
- Analysis of class imbalance

The analysis showed that **Torque**, **Rotational Speed**, and **Tool Wear** were particularly relevant for failure prediction.

## 🤖 Models

Three classification models were compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Because machine failures are rare, the models were evaluated using metrics beyond accuracy, especially:

- Precision
- Recall
- F1-score
- Precision-Recall curve
- Average Precision

## 🌲 Random Forest Optimization

The Random Forest was optimized using `GridSearchCV` with 5-fold cross-validation.

Best hyperparameters:

```python
{
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "n_estimators": 200
}
```

Cross-validation F1 score:

```text
0.736
```

## 📊 Final Performance

Performance on the test set for the failure class:

| Metric | Score |
|---|---:|
| Precision | 0.84 |
| Recall | 0.71 |
| F1-score | 0.77 |
| Average Precision | 0.798 |

Confusion matrix:

```text
[[1923    9]
 [  20   48]]
```

The model correctly identified 48 of the 68 failures in the test set while generating only 9 false alarms.

## 🔍 Feature Importance

| Feature | Importance |
|---|---:|
| Torque | 33.6% |
| Rotational Speed | 28.2% |
| Tool Wear | 21.0% |
| Air Temperature | 9.7% |
| Process Temperature | 6.1% |
| Machine Type | 1.3% |

Torque, rotational speed, and tool wear account for approximately **82.8%** of the Random Forest impurity-based feature importance.

## 🖥️ Streamlit Application

The trained model is integrated into a Streamlit application.

Users can enter machine operating conditions and obtain:

- A failure / normal prediction
- Estimated failure probability
- A simple visualization of the predicted risk

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Git / GitHub

## 📁 Project Structure

```text
machine-failure-prediction/
│
├── app.py
├── machine_failure_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Run Locally

Clone the repository:

```bash
git clone <repository-url>
cd machine-failure-prediction
```

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📌 Key Takeaways

This project demonstrates an end-to-end tabular machine learning workflow:

**EDA → preprocessing → baseline modeling → class imbalance handling → model comparison → cross-validation → hyperparameter tuning → evaluation → deployment**

A key lesson from the project is that **accuracy alone is not sufficient for highly imbalanced classification problems**. Precision, recall, F1-score, and Precision-Recall analysis provide a much more meaningful evaluation of failure detection performance.
