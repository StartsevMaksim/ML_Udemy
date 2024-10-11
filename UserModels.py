import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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

        X_train = np.insert(X_train, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0      
        prev_coeffs = np.random.rand(len(X_train[0]))
        self.reg_coeffs = 2 * prev_coeffs
        while np.linalg.norm(self.reg_coeffs-prev_coeffs, ord=2) > self.accurate:
            next_approximation = prev_coeffs - self.train_speed * countGradient(X_train, y_train, prev_coeffs)
            prev_coeffs = self.reg_coeffs
            self.reg_coeffs = next_approximation
        
    def predict(self, X_test):
        try:
            X_test = np.insert(X_test, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0
            return X_test @ self.reg_coeffs
        except:
            print('Не выполнено обучение')

#Эластичная сетка
class UElasticNet:
    def __init__(self, alpha=None, l1_ratio=None, train_speed=0.001, accurate=0.000001):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.train_speed = train_speed
        self.accurate = accurate

    def _countGradient(self, X_train, y_train):
        vector_1 = X_train.T @ y_train
        vector_2 = X_train.T @ (X_train @ self.coeffs_)
        vector_3 = np.append([0], self.coeffs_[1:])
        vector_4 = np.append([0], np.ones(len(self.coeffs_) - 1))
        return ((1 / len(y_train)) * (vector_2 - vector_1)
                + self.alpha * ((1 - self.l1_ratio) * vector_3
                                 + self.l1_ratio * vector_4))
        
    def fit(self, X_train, y_train):
        try:
            if self.alpha is None or self.l1_ratio is None:
                raise ValueError()
            X_train = np.insert(X_train, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0  
            y_train = np.array(y_train)
            self.coeffs_ = np.zeros(len(X_train[0]))
            next_coeffs = np.random.rand(len(X_train[0]))
            while np.linalg.norm(next_coeffs - self.coeffs_, ord=2) > self.accurate:
                self.coeffs_ = next_coeffs
                next_coeffs = self.coeffs_ - self.train_speed * self._countGradient(X_train, y_train)
            self.coeffs_ = next_coeffs
        except ValueError:
            print('Не заданы параметры модели')
        
    def predict(self, X_test):
        try:
            X_test = np.insert(X_test, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0
            return X_test @ self.coeffs_
        except:
            print('Не выполнено обучение')

#Логистическая регрессия
class ULogisticRegression:
    def __init__(self, train_speed=0.1, accurate=0.0001):
        self.train_speed = train_speed
        self.accurate = accurate

    def _countGradient(self, X_train, y_train):
        vector_1 = 1 / (1 + np.exp(- X_train @ self.coeffs_))
        return (1 / len(X_train)) * (X_train.T @ (vector_1 - y_train))
        
    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0  
        y_train = np.array(y_train)
        self.coeffs_ = np.random.rand(len(X_train[0]))
        next_coeffs = np.zeros(len(X_train[0]))
        while np.linalg.norm(next_coeffs - self.coeffs_, ord=2) > self.accurate:
            self.coeffs_ = next_coeffs
            next_coeffs = self.coeffs_ - self.train_speed * self._countGradient(X_train, y_train)
        self.coeffs_ = next_coeffs
        
    def predict(self, X_test):
        try:
            X_test = np.insert(X_test, 0, 1, axis=1) # Свободный коэффициент имеет индекс 0
            probability = 1 / (1 + np.exp(- X_test @ self.coeffs_))
            return np.array([0 if prob < 0.5 else 1 for prob in probability])
        except:
            print('Не выполнено обучение')

#Поиск по сетке с кросс-валидацией
class UGridSearchCV:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv

    def _builtParamsSet(self, param_grid, keys, key_index, result):
        if key_index < len(keys):
            for value in param_grid[keys[key_index]]:
                result = dict(result)
                result[keys[key_index]] = value
                yield from self._builtParamsSet(param_grid, 
                                                keys, 
                                                key_index+1, 
                                                result)
        else:
            yield result

    def _getTrainTest(self, X, y, test_indexes):
        test_indexes = set(test_indexes)
        X_train, y_train, X_test, y_test = [], [], [], []
        for index in range(len(X)):
            if index in test_indexes:
                X_test.append(X[index])
                y_test.append(y[index])
            else:
                X_train.append(X[index])
                y_train.append(y[index])
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.best_estimator_ = None
        self.best_params_ = None
        self._X_train_best = None
        self._y_train_best = None
        lowest_error = None
        for test_indexes in np.array_split(np.random.permutation(len(X)), self.cv):
            X_train, y_train, X_test, y_test = self._getTrainTest(X, 
                                                                  y,
                                                                  test_indexes)
            for params in self._builtParamsSet(self.param_grid, 
                                               list(self.param_grid.keys()), 
                                               0, 
                                               dict()):
                for param_name, param_value in params.items():
                    self.estimator.__setattr__(param_name, param_value)
                self.estimator.fit(X_train, y_train)
                y_predict = self.estimator.predict(X_test)
                MSE = mean_squared_error(y_test, y_predict)
                if (lowest_error is None) or (MSE < lowest_error):
                    self._X_train_best = np.array(X_train)
                    self._y_train_best = np.array(y_train)
                    self.best_params_ = dict(params)
                    lowest_error = MSE
                print('----[CV] END .......', params, 'MSE =', MSE)
        
        for param_name, param_value in self.best_params_.items():
            self.estimator.__setattr__(param_name, param_value)
        self.estimator.fit(self._X_train_best, self._y_train_best)
        self.best_estimator_ = self.estimator