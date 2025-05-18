import pandas as pd
import numpy as np

df = pd.read_csv('materials.csv')

#Compute the correlation coefficient
correlation_time = np.corrcoef(df['Strength'], df['Time'])[0, 1]
correlation_pressure = np.corrcoef(df['Strength'], df['Pressure'])[0, 1]
correlation_temperature = np.corrcoef(df['Strength'], df['Temperature'])[0, 1]

print(f"Correlation coefficient between Strength and Time: {correlation_time}")
print(f"Correlation coefficient between Strength and Pressure: {correlation_pressure}")
print(f"Correlation coefficient between Strength and Temperature: {correlation_temperature}")

#Multiple Linear Regression
X = df[['Time', 'Pressure', 'Temperature']].values
y = df['Strength'].values

X = np.hstack((np.ones((X.shape[0], 1)), X))

#Compute the coefficients
X_transpose = X.T
X_transpose_dot_X = np.dot(X_transpose, X)
X_transpose_dot_y = np.dot(X_transpose, y)
coefficients = np.linalg.inv(X_transpose_dot_X).dot(X_transpose_dot_y)

#Predict Strength
data_points = np.array([[32.1, 37.5, 128.95], [36.9, 35.37, 130.03]])
data_points = np.hstack((np.ones((data_points.shape[0], 1)), data_points))
predicted_strength = np.dot(data_points, coefficients)

print(f"\nPredicted Strength for data point [32.1, 37.5, 128.95]: {predicted_strength[0]}")
print(f"Predicted Strength for data point [36.9, 35.37, 130.03]: {predicted_strength[1]}")
