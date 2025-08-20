import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("🏡 California Housing Price Prediction using Linear Regression")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Feature engineering (same as before)
safe_rooms = df["AveRooms"].replace(0, np.nan)
safe_occup = df["AveOccup"].replace(0, np.nan)

df["PopDensity"] = df["Population"] / (df["AveOccup"] + 1e-5)
df["IncomePerHouseAge"] = df["MedInc"] / (df["HouseAge"] + 1)
df["BedroomsPerRoom"] = df["AveBedrms"] / (safe_rooms + 1e-5)
df["PeoplePerHousehold"] = df["Population"] / (safe_occup + 1e-5)

df = df.fillna(0)

st.subheader("Dataset Preview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Evaluation Results")
st.write("Root Mean Squared Error (RMSE):", rmse)
st.write("R² Score:", r2)

st.subheader("Sample Predictions vs Actual")
st.dataframe(pd.DataFrame({"Actual": y_test[:10].values, "Predicted": y_pred[:10]}))

st.subheader("📈 Actual vs Predicted Visualization")
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(y_test, y_pred, alpha=0.5, color="blue")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        color="red", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual House Value")
ax.set_ylabel("Predicted House Value")
ax.set_title("Actual vs Predicted House Prices")
ax.legend()
st.pyplot(fig)

st.subheader("Feature Importance (Coefficients)")
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)
st.dataframe(coeff_df)
