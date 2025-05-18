import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('avgHigh_jan_1895-2018.csv')

years = df.iloc[:, 0].values.reshape(-1, 1)
temperatures = df.iloc[:, 1].values

test_size = 0.2

# Split the dataset into training and test sets
split_index = int(len(years) * (1 - test_size))
X_train, X_test = years[:split_index], years[split_index:]
y_train, y_test = temperatures[:split_index], temperatures[split_index:]

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

#Predict temperature 
y_pred = model.predict(X_test)

#Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Actual temperatures from the test dataset:")
print(y_test)
print("\nPredicted temperatures:")
print(y_pred)
print(f"\nRoot Mean Square Error (RMSE): {rmse}")

#plotting
plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='orange', label='Test')
plt.plot(X_train, model.predict(X_train), color='red', label='Model')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Slope: 0.00, Intercept: 25.37, Test size: 0.15 (19/124), RMSE: 4.62')
plt.legend()
plt.grid(True)
plt.show()
