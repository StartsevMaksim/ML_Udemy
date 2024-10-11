import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

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