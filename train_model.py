import pandas as pd
import sklearn.datasets
from xgboost import XGBRegressor
import pickle

dataset = sklearn.datasets.fetch_california_housing()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
Y = dataset.target

model = XGBRegressor()
model.fit(X, Y)

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as house_price_model.pkl")
