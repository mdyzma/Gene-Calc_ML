import numpy as np

class Validation():

    def __init__(self, X, y, model_type):
        self.X_array = X
        self.y_vector = y
        self.model_type = model_type

    def shape_validation(self):
        """Method to check in-data shape"""

        X_shape = self.X_array.shape
        y_shape = self.y_vector.shape

        assert X_shape[1] < 10, "More then allowed predictors !"
        assert y_shape[0] < 10000, "More then allowed records !"
        print("In data shape is accurate")

    def data_quality(self):
        """method to find NaN"""

        if np.isnan(self.X_array).any() == True:
            print("NaN in X_array data !")
        
        else:
            print("NaN no in X_array")

        if np.isnan(self.y_vector).any() == True:
            print("NaN in y_vector data !")

        else:
            print("NaN no in y_vector")