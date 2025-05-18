import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

df = pd.read_csv('materials.csv')

strength = df['Strength'].values
time = df['Time'].values
pressure = df['Pressure'].values
temperature = df['Temperature'].values

def correlation(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    stddev_x = (sum((xi - mean_x) ** 2 for xi in x) / len(x)) ** 0.5
    stddev_y = (sum((yi - mean_y) ** 2 for yi in y) / len(y)) ** 0.5
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
    return covariance / (stddev_x * stddev_y)

#correlation coefficients
correlation_time = correlation(strength, time)
correlation_pressure = correlation(strength, pressure)
correlation_temperature = correlation(strength, temperature)

features = ['Time', 'Pressure', 'Temperature']
correlations = [correlation_time, correlation_pressure, correlation_temperature]
sorted_features = [feature for _, feature in sorted(zip(correlations, features), key=lambda pair: abs(pair[0]), reverse=True)]
top_features = sorted_features[:2]

# Swap the pressure and temperature axis
if 'Pressure' in top_features and 'Temperature' in top_features:
    top_features[0], top_features[1] = top_features[1], top_features[0]

X = df[top_features].values
y = strength

#Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)

#meshgrid for 3D plot
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
X_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
y_mesh_pred = model.predict(X_mesh).reshape(x1_mesh.shape)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh_pred, color='blue', alpha=0.5)
ax.scatter(X[:, 0], X[:, 1], y, color='red')

ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel('Strength')
ax.set_title('3D Graph')

plt.show()
