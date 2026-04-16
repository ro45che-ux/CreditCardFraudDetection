# Credit Card Fraud Detection 

## Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning.
The dataset is highly imbalanced, making it a real-world classification problem.

---

##  Features

* Data preprocessing and splitting
* Handling imbalanced dataset using **SMOTE**
* Model training using **Logistic Regression**
* Evaluation using **Precision, Recall, F1-score**
* Model saving using `joblib`

---

##  Key Concepts

* Imbalanced Data Handling
* Precision vs Recall Trade-off
* SMOTE (Synthetic Minority Oversampling Technique)
* Classification Metrics

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn

---

##  Project Structure

```
credit-card-fraud-detection/
│
├── data/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
├── models/
│   └── model.pkl
├── main.py
└── README.md
```

---

##  How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib
python3 main.py
```

---

##  Dataset

Dataset is too large to upload on GitHub.
Download from:
 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file inside:

```
data/creditcard.csv
```

---

##  Results

* Before SMOTE:

  * Low recall for fraud detection
* After SMOTE:

  * Significant improvement in recall
  * Trade-off with precision

---

##  Conclusion

SMOTE helps improve fraud detection by balancing the dataset,
but introduces more false positives.
This highlights the importance of choosing the right evaluation metric.

---

##  Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Hyperparameter tuning
* ROC-AUC visualization
* Deployment (Flask/Streamlit)

---
