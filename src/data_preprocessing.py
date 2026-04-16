import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)