import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.loss_historial=[]
        self.valores_w=[]
        self.valores_b=[]

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        #AQUI YA SE HA AÑADIDO UNA COLUMNA PARA EL INTERCEPTO

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session


        # X_b = np.c_[np.ones((X.shape[0], 1)), X] #esto ya se ha hecho antes en la funcion que llama a esto
    
        w= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) #esta en la formula de los parametros optimos para el caso multivariante

        
        self.intercept = w[0] #el primer valor corresponde al intercepto
        self.coefficients = w[1:]

    
        # Store the intercept and the coefficients of the model
        

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000, plot_gradient=False):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01


        
        
        # Implement gradient descent (TODO)


        for epoch in range(iterations):
            predictions = self.predict(X)
            #las predicciones son un vector con las mismas columnas [2,6,4,7,4] (5), como filas hay en el vector de X
        

            error = predictions - y
            
            # TODO: Write the gradient values and the updates for the paramenters
    
            gradient = (1/m) * X.T@(error)


            self.loss_historial.append(np.sum(error**2))
            self.valores_w.append(self.coefficients.copy())  # Suponiendo 1 coeficiente
            self.valores_b.append(self.intercept)

            # Separar actualización para evitar errores de dimensiones
            self.intercept -= learning_rate * gradient[0]  #Actualiza el intercepto
            #en el intercepto la X va a ser 1 siempre
            self.coefficients -= learning_rate * gradient[1:]  #Actualiza la pendiente


            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = np.sum(error**2)
                print(f"Epoch {epoch}: MSE = {mse}")


    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
                raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions =self.intercept + X*self.coefficients
        else:
            # TODO: Predict when X is more than one variable
            
        
            predictions = self.intercept+X[:,1:]@self.coefficients
            #he puesto esto X[:,1:], porque en la funcion fit, que es la primera a la que se invoca, los datos
            #Xse convierten en X with bias
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    rss=np.sum((y_true-y_pred)**2)#con respecto la predicciom
    tss=np.sum((y_true-np.mean(y_true))**2)
    r_squared=1-(rss/tss)


    # Root Mean Squared Error
    # TODO: Calculate RMSE

    rmse = np.sqrt(np.mean((y_true-y_pred)**2))


    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(np.abs(y_true-y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """


    #lo que hace one hot encode es recorrer cada 
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        
        # TODO: Extract the categorical column
        categorical_column = X[:,index]

        # TODO: Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([[1 if val == category else 0 for category in unique_values] for val in categorical_column])

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        
        X_transformed = np.delete(X_transformed, index, axis=1)
        
        X_transformed = np.hstack((X_transformed[:, :index], one_hot, X_transformed[:, index:]))
        # divide X_transformed en dos partes:
        # Las columnas antes de la columna categórica (:index).
        # Inserta la nueva matriz one_hot.
        # Mantiene las columnas después de la original (index:), sin incluir la original.


    return X_transformed


