from src.data_preprocessing import load_data, preprocess_data, split_data, apply_smote
from src.train_model import train_model, save_model
from src.evaluate import evaluate_model

# 1. Load data
df = load_data("data/creditcard.csv")

# 2. Preprocess
X, y = preprocess_data(df)

# 3. Split
X_train, X_test, y_train, y_test = split_data(X, y)

# 4. Apply SMOTE
X_train_sm, y_train_sm = apply_smote(X_train, y_train)

# 5. Train model
model = train_model(X_train_sm, y_train_sm)

# 6. Evaluate
evaluate_model(model, X_test, y_test)

# 7. Save model
save_model(model, "models/model.pkl")