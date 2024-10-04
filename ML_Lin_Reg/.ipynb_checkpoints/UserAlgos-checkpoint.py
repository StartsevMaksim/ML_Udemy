import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

#Простая линейная регрессия
class USimpleLinearRegression:
    def __init__(self):
        self.B0 = None
        self.B1 = None

    def fit(self, x_train, y_train):
        self.B1 = ((x_train - x_train.mean()) @ (y_train - y_train.mean())) / ((x - x.mean()) @ (x - x.mean()))
        self.B0 = y_train.mean() - self.B1 * x_train.mean()

    def predict(self, x_test):
        try:
            return self.B1 * x_test + self.B0
        except:
            print('Не выполнено обучение')

#Многомерная линейная регрессия
class ULinearRegression:
    def __init__(self, train_speed=0.001, accurate=0.000001):
        self.train_speed = train_speed
        self.accurate = accurate
        self.reg_coeffs = None
        
    def fit(self, X_train, y_train):
        def countGradient(X_train, y_train, reg_coeffs):
            vector_1 = X_train.T @ y_train
            vector_2 = X_train.T @ (X_train @ reg_coeffs)
            return (1 / len(y_train)) * (vector_2 - vector_1)

        X_train = scale(X_train)
        X_train = np.insert(X_train, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0      
        prev_coeffs = np.random.rand(len(X_train[0]))
        self.reg_coeffs = 2 * prev_coeffs
        while np.linalg.norm(self.reg_coeffs-prev_coeffs, ord=2) > self.accurate:
            next_approximation = prev_coeffs - self.train_speed * countGradient(X_train, y_train, prev_coeffs)
            prev_coeffs = self.reg_coeffs
            self.reg_coeffs = next_approximation
        
    def predict(self, X_test):
        try:
            X_test = scale(X_test)
            X_test = np.insert(X_test, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0
            return X_test @ self.reg_coeffs
        except:
            print('Не выполнено обучение')

#Создание полиномиальный коэффициентов 2ой степени
class UPolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def transform(self, X):
        polynomial_X = pd.DataFrame(X)
        for i_index in range(len(X.columns)):
            for j_index in range(i_index, len(X.columns)):
                polynomial_X.insert(len(polynomial_X.columns), 
                                    str(i_index) + str(j_index), 
                                    X.iloc[:, i_index] * X.iloc[:, j_index])
        return np.array(polynomial_X)

#Масштабирование признаков
    #Стандартизация
class UStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        try:
            for row_index in range(len(X)):
                for col_index in range(len(X[row_index])):
                    X[row_index][col_index] = (X[row_index][col_index] - self.mean[col_index]) / self.std[col_index]
            return X
        except:
            print('Не выполнено обучение')