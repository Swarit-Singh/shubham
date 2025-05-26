import joblib
from sklearn.ensemble import RandomForestRegressor
from ml_training.data_loader import load_threshold_dataset

# 1. Load features & targets
X, y = load_threshold_dataset()

# 2. Train regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X, y)

# 3. Save
joblib.dump(reg, 'models/ml_assisted/thres_regressor.pkl')
print("Saved threshold regressor to models/ml_assisted/thres_regressor.pkl")

