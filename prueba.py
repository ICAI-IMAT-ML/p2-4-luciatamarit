# Import here whatever you may need
import numpy as np
import pytest
import numpy as np
from src.Lab_2_4_LR2 import (
    LinearRegressor,
    evaluate_regression,
    one_hot_encode,
)  # Assuming the class is in linear_regressor.py


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
X_with_bias = np.insert(
            X, 0, 1, axis=1
        ) 
X=X_with_bias
print(X_with_bias[:,1:])
coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
intercept = np.random.rand() * 0.01
print(coefficients,intercept)
print(intercept+X[:,1:]@coefficients)


# # X = np.array([1, 2, 3])
# # print(np.ndim(X))

# if np.ndim(X) == 1:
#      X = X.reshape(-1, 1)

# # print(X)
# model = LinearRegressor()
# model.fit(X, y, method="gradient_descent", learning_rate=0.01, iterations=10000)
X = np.array(
        [
            ["Red", 10],
            ["Blue", 20],
            ["Green", 30],
            ["Red", 40],
        ],
        dtype=object,
    )


# print(X[:,1:])
categorical_indices = [0]

# # one_hot_encode(X, categorical_indices, drop_first=False)
# categorical_column = X[:,0]
# unique_values = np.unique(categorical_column)
# # print(unique_values)
# # one_hot = [[1 if unique_values[i]==valor else 0 for i in range(len(unique_values))] for valor in unique_values ]
# # print(one_hot)



# # numerical_column = X[:, 1].astype(int).reshape(-1, 1)
# # print(numerical_column)
# # result=numerical_column
# # # result = np.hstack((numerical_column, one_hot))
# # # print(result)
# # for i in range(len(numerical_column)):
# #     # print(i)


# #     result=np.insert(result,i,one_hot)

# # import numpy as np

def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. 
    This function supports string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    
    for index in sorted(categorical_indices, reverse=True):
        # Extraer la columna categórica
        categorical_column = X[:, index]
        
        # Obtener categorías únicas
        unique_values = np.unique(categorical_column)

        # Crear matriz One-Hot
        one_hot = np.array([[1 if val == category else 0 for category in unique_values] for val in categorical_column])
        

        # Si drop_first es True, eliminar la primera columna de one_hot
        if drop_first:
            one_hot = one_hot[:, 1:]

        # Eliminar la columna categórica original
        X_transformed = np.delete(X_transformed, index, axis=1)

        # Insertar la matriz One-Hot en X_transformed
        
        X_transformed = np.hstack((one_hot, X_transformed))
        

    return X_transformed





