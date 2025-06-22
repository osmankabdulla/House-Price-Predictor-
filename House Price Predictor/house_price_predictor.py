import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Create a synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    "sqft": np.random.randint(500, 3500, 100),
    "bedrooms": np.random.randint(1, 5, 100),
    "bathrooms": np.random.randint(1, 4, 100),
})

# Generate price using a basic formula + noise
data["price"] = (
    data["sqft"] * 300 +
    data["bedrooms"] * 5000 +
    data["bathrooms"] * 10000 +
    np.random.normal(0, 10000, 100)
)

# Step 2: Prepare data
X = data[["sqft", "bedrooms", "bathrooms"]]
y = data["price"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
print(f"Root Mean Squared Error: {rmse:.2f}")

# Step 6: Save model
joblib.dump(model, "house_price_model.pkl")
print("Trained model saved to 'house_price_model.pkl'")