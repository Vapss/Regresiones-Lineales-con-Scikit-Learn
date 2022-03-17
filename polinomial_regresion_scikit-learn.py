import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator
import pandas as pd
from  sklearn import preprocessing
from sklearn.linear_model import SGDRegressor

mse_list = []
r2_list = []

#Generación de los datos
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)
#~ print ('x{}\ny{}'.format(x,y))
plt.scatter(x,y, s=10)

#Se agrega una nueva dimensión para que variable sea un arreglo
#~ x = np.reshape(x, (-1,1))
x = x[:, np.newaxis]
#~ print ('x{}\ny{}'.format(x,y))

#Modelo de regresión lineal
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
plt.plot(x, y_pred, color='r')

#Cálculo del error cuadrado medio y r2
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print ('Regresión lineal\nmse: {} r2: {}'.format(mse, r2))
mse_list.append(mse)
r2_list.append(r2)

#Conversión de las variables de la ecuación original a polinomio de grado 2
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)
print (x_poly)

#~ #Modelo de regresión polinomial
model_poly = LinearRegression()
model_poly.fit(x_poly, y)
y_poly_pred = model_poly.predict(x_poly)
#~ print('y_poly_pred {}'.format(y_poly_pred))

#~ #Cálculo del error cuadrado medio y r2
mse = mean_squared_error(y, y_poly_pred)
r2 = r2_score(y, y_poly_pred)
print ('Regresión polinomial grado 2\nmse: {} r2: {}'.format(mse, r2))
mse_list.append(mse)
r2_list.append(r2)

#~ plt.plot(x, y_poly_pred, color='g')
#~ print(pd.DataFrame({'x': np.reshape(x,(1,-1))[0], 'Predicted': y_poly_pred}))

#~ #Ajustes para que la curva trazada se vea correctamente
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
#~ print (tuple(sorted_zip))
x_sorted, y_poly_pred = zip(*sorted_zip)
#~ print(pd.DataFrame({'x': np.reshape(x_sorted,(1,-1))[0], 'Predicted': y_poly_pred}))
plt.plot(x_sorted, y_poly_pred, color='g')

# Modelo de regresión polinomial grado 3
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)
model_poly = LinearRegression()
model_poly.fit(x_poly, y)
y_poly_pred = model_poly.predict(x_poly)
mse = mean_squared_error(y, y_poly_pred)
r2 = r2_score(y, y_poly_pred)
print ('Regresión polinomial grado 3\nmse: {} r2: {}'.format(mse, r2))
mse_list.append(mse)
r2_list.append(r2)

#Ajustes para que la curva trazada se vea correctamente
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
print (tuple(sorted_zip))
x_sorted, y_poly_pred = zip(*sorted_zip)
print(pd.DataFrame({'x': np.reshape(x_sorted,(1,-1))[0], 'Predicted': y_poly_pred}))
plt.plot(x_sorted, y_poly_pred, color='b')

#~ ####Escalado de los datos####
#~ #Standard Scaler
x_poly_standard_scaler = preprocessing.StandardScaler().fit_transform(x_poly)
print (x_poly_standard_scaler)
model_poly.fit(x_poly_standard_scaler, y)
y_poly_pred = model_poly.predict(x_poly_standard_scaler)
mse = mean_squared_error(y, y_poly_pred)
r2 = r2_score(y, y_poly_pred)
print ('Regresión polinomial grado 3 escalado estándar\nmse: {} r2: {}'.format(mse, r2))
mse_list.append(mse)
r2_list.append(r2)

x_poly_robust_scaler = preprocessing.RobustScaler().fit_transform(x_poly)
model_poly.fit(x_poly_robust_scaler, y)
y_poly_pred = model_poly.predict(x_poly_robust_scaler)
mse = mean_squared_error(y, y_poly_pred)
r2 = r2_score(y, y_poly_pred)
print ('Regresión polinomial grado 3 escalado robusto\nmse: {} r2: {}'.format(mse, r2))
mse_list.append(mse)
r2_list.append(r2)

regr = SGDRegressor(learning_rate = 'constant', eta0 = 0.001, max_iter= 10000)
regr.fit(x_poly_robust_scaler, y)
y_poly_pred = regr.predict(x_poly_robust_scaler)
mse = mean_squared_error(y, y_poly_pred)
r2 = r2_score(y, y_poly_pred)
print ('Regresión polinomial estocástico grado 3 escalado robusto\nmse: {} r2: {}'.format(mse, r2))

# initialize data of lists.
data = {'mse':mse_list,
        'r2':r2_list}
 
# Creates pandas DataFrame.
df = pd.DataFrame(data, index =['Regresión lineal',
                                'Regresión polinomial grado 2',
                                'Regresión polinomial grado 3',
                                'Regresión polinomial grado 3 escalado estándar',
                                'Regresión polinomial grado 3 escalado robusto'])

print(df)
#~ plt.show()
