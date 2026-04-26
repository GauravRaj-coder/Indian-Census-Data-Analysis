# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 2. LOAD DATA
file_path = r"C:\Users\gaura\OneDrive\Desktop\PYTHON PROJECT\A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA (1).csv"
df = pd.read_csv(file_path, low_memory=False)
print("Original Shape:", df.shape)
# 3. REMOVE EXTRA HEADER ROWS
df = df.iloc[6:].reset_index(drop=True)
# 4. DROP EMPTY COLUMNS
df = df.dropna(axis=1, how='all')
# 5. CONVERT TO NUMERIC
numeric_df = df.apply(pd.to_numeric, errors='coerce')
# 6. REMOVE BAD DATA (SMART CLEANING)
numeric_df = numeric_df.dropna(axis=1, thresh=len(numeric_df)*0.3)
numeric_df = numeric_df.dropna(axis=0, thresh=numeric_df.shape[1]*0.3)

print("After Cleaning Shape:", numeric_df.shape)
# 7. FILL MISSING VALUES
numeric_df = numeric_df.fillna(numeric_df.mean())
# 8. CHECK DATA
if numeric_df.shape[0] < 10 or numeric_df.shape[1] < 2:
    print("❌ Not enough usable data")
    exit()

print("\nCleaned Data Preview:")
print(numeric_df.head())
# 9. EDA
print("\nStatistical Summary:")
print(numeric_df.describe())

print("\nCorrelation Matrix:")
corr = numeric_df.corr()
print(corr)
# 10. HEATMAP (MANUAL)
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
# 11. HISTOGRAMS
numeric_df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.show()
# 12. SCATTER PLOT
plt.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter Plot")
plt.show()
# 13. LINEAR REGRESSION
X = numeric_df.iloc[:, :-1]
y = numeric_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
# 14. PREDICTIONS
y_pred = model.predict(X_test)
# 15. ACTUAL VS PREDICTED TABLE
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))
# 16. LINE PLOT COMPARISON
plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Line Plot")
plt.show()
# 17. SCATTER COMPARISON
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Scatter")
plt.show()
# 18. RESIDUAL ANALYSIS
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
# 19. MODEL EVALUATION
print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

