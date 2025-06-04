import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#LOADING OF THE DATASET 

data= pd.read_csv("train.csv")

#NOW FOR THE SELECTING FEATURES , WHICH WERE DESCRIBED 
"""
1. GrLivArea
2. BedroomAbvGr
3. FullBath

"""
features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = data['SalePrice']

#SPLITIING INTO TRAINING AND TESTING SETS
X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#THE FINALLY CREATION OF MODEL 
model = LinearRegression()
model.fit(X_train,y_train)

#MAKING PREDICTIONS 

y_pred = model.predict(X_test)

#EVALUATION PROCESS

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Model Coefficients:")
print(f" - Square Footage Coefficient (GrLivArea): {model.coef_[0]:.2f}")
print(f" - Bedrooms Coefficient (BedroomAbvGr): {model.coef_[1]:.2f}")
print(f" - Bathrooms Coefficient (FullBath): {model.coef_[2]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.6, color='green', edgecolor='black')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()