# Churn-project
# 📉 Customer Churn Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-red?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

An end-to-end Machine Learning project that predicts whether a customer is likely to churn using multiple classification algorithms and a production-ready prediction pipeline.

The project follows a complete Machine Learning workflow from data preprocessing and exploratory analysis to model comparison, evaluation, and deployment-ready pipeline generation.

# 📌 Problem Statement

Customer churn is one of the biggest challenges faced by subscription-based businesses such as telecom, banking, insurance, and SaaS companies.

The goal of this project is to build an intelligent classification model capable of predicting whether a customer is likely to leave the company, allowing businesses to take proactive retention measures.

# ✨ Features
- Complete Data Cleaning Pipeline
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Train / Validation / Test Split
- Scikit-Learn Pipeline
- Multiple Machine Learning Models
- Model Performance Comparison
- Saved Preprocessing Pipeline
- Saved Production Model
- Streamlit Web Application
- Reusable Prediction Pipeline

# 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Models | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Model Saving | Joblib |
| Web App | Streamlit |
| Others | JSON, Pickle |

# 📂 Project Structure

```
Churn-project
│
├── artifacts/
│   ├── confusion matrices
│   ├── model comparison
│   ├── metrics
│   ├── metadata
│   └── preprocessing files
│
├── data/
│   ├── raw dataset
│   ├── cleaned dataset
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── models/
│   ├── Logistic Regression
│   ├── Random Forest
│   ├── XGBoost
│   ├── LightGBM
│   ├── Best Model
│   └── Final Prediction Pipeline
│
├── notebooks/
│   ├── EDA
│   ├── Model Training
│   └── Prediction Pipeline
│
├── utils/
│
├── streamlit_churn_app.py
├── requirements.txt
└── README.md
```

---

# 🔄 Machine Learning Workflow

```
Customer Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Model Comparison
        │
        ▼
Best Model Selection
        │
        ▼
Save Pipeline
        │
        ▼
Streamlit Prediction App
```

---

# 📊 Exploratory Data Analysis

The project includes detailed exploratory analysis to understand customer behaviour.

Some analyses performed include:

- Missing Value Analysis
- Duplicate Detection
- Customer Distribution
- Churn Distribution
- Numerical Feature Analysis
- Categorical Feature Analysis
- Correlation Analysis
- Outlier Detection

---

# ⚙️ Data Preprocessing

The preprocessing pipeline consists of:

- Handling Missing Values
- Encoding Categorical Variables
- Feature Scaling
- Feature Transformation
- Train-Test Split
- Pipeline Serialization

The preprocessing pipeline is stored as:
```
preprocessor.pkl
```
---
# 🤖 Models Trained
The following classification models were implemented and compared.
| Model |
|---------|
| Logistic Regression |
| Random Forest |
| XGBoost |
| LightGBM |
Each model was evaluated using the same preprocessing pipeline to ensure a fair comparison.
---
# 📈 Evaluation Metrics
The models were compared using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

Performance reports are stored inside the **artifacts** directory.
---
# 🏆 Best Model
After evaluating multiple machine learning algorithms, the best-performing model was selected and exported as a reusable prediction pipeline.
Saved Files
```
best_model_pipeline.pkl
final_churn_pipeline.pkl
```
These files can directly predict customer churn without retraining.

# Streamlit Application

The project also includes an interactive Streamlit application where users can provide customer information and receive churn predictions in real time.

To run locally:

```bash
git clone https://github.com/Parth-Jos-hi/Churn-project.git

cd Churn-project

pip install -r requirements.txt

streamlit run streamlit_churn_app.py
```
# Generated Artifacts
The project automatically saves:

- Trained Models
- Feature Names
- Metadata
- Confusion Matrices
- Performance Reports
- Comparison Tables
- Preprocessing Pipeline

These artifacts make the project reproducible and deployment-ready.

# 🚀 Future Improvements

- Hyperparameter Tuning
- SHAP Explainability
- FastAPI Deployment
- Docker Support
- CI/CD Pipeline
- Cloud Deployment
- Model Monitoring

# What I Learned
This project helped strengthen my understanding of:

- Machine Learning Pipelines
- Feature Engineering
- Classification Algorithms
- Model Comparison
- Model Serialization
- Production-ready Workflow
- Streamlit Deployment
- End-to-End ML Project Development
# 👨‍💻 Author
**Parth Joshi**
AI & Machine Learning Undergraduate
GitHub:
https://github.com/Parth-Jos-hi
---

⭐ If you found this project useful, consider giving it a star!
