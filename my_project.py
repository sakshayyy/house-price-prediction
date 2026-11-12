import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# STEP 1: Create Dataset
# -------------------------------
data = {
    "Area": [
        1000,1200,1500,1800,2000,2200,2500,2800,3000,3500,
        900,1100,1300,1600,1700,1900,2100,2400,2600,3200
    ],
    "Bedrooms": [
        2,2,3,3,4,4,4,5,5,6,
        2,2,3,3,3,4,4,4,5,5
    ],
    "Age": [
        10,8,7,6,5,4,3,2,2,1,
        12,9,8,7,6,5,4,3,2,1
    ],
    "Price": [
        3000000,3400000,4200000,5000000,5800000,
        6200000,7000000,8200000,9000000,10500000,
        2800000,3200000,3700000,4500000,4800000,
        5500000,6100000,6900000,7800000,9500000
    ]
}

df = pd.DataFrame(data)

print("\nDataset:")
print(df)

# -------------------------------
# STEP 2: Features and Target
# -------------------------------
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# -------------------------------
# STEP 3: Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 4: Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# STEP 5: Predict Test Data
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# STEP 6: Evaluate Model
# -------------------------------
print("\nModel Evaluation")
print("-" * 30)
print("R2 Score:", round(r2_score(y_test, y_pred), 4))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("Root Mean Squared Error:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# -------------------------------
# STEP 7: Predict New House
# -------------------------------
print("\nPredict House Price")
print("-" * 30)

area = float(input("Enter Area (sq.ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
age = int(input("Enter House Age (years): "))

new_house = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Age": [age]
})

predicted_price = model.predict(new_house)

print("\nEstimated House Price = ₹{:,.2f}".format(predicted_price[0]))

# -------------------------------
# STEP 8: Model Coefficients
# -------------------------------
print("\nModel Coefficients")
print("-" * 30)
print("Area Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Age Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)

# -------------------------------
# STEP 9: Compare Actual vs Predicted
# -------------------------------
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.astype(int)
})

print("\nActual vs Predicted Prices")
print(comparison)

# -------------------------------
# STEP 10: Plot Graph
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.grid(True)
plt.show()