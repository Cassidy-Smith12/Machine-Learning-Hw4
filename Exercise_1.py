import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('avgHigh_jan_1895-2018.csv')

years = df.iloc[:, 0].values.reshape(-1, 1)
temperatures = df.iloc[:, 1].values

#Linear Regression
model = LinearRegression().fit(years, temperatures)

#predict temperatures
future_years = np.array([201901, 202301, 202401]).reshape(-1, 1)
predicted_temperatures = model.predict(future_years)

#predicted
predicted_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Temperature': predicted_temperatures})
print(predicted_df)

#plot
plt.figure(figsize=(12, 8))
plt.scatter(years, temperatures, color='blue', label='Data Points')
plt.scatter(future_years, predicted_temperatures, color='green', label='Prediced')
plt.plot(years, model.predict(years), color='red', label='Fitted Line')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('January Average High Temperatures. Slope: 0.00, Intercept: 8.69')
plt.legend()
plt.grid(True)
plt.show()
