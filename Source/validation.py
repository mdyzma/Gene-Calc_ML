import numpy as np

class Validation():

    def __init__(self, in_data, model_type):
        self.data = in_data
        self.model = model_type


    def data_val_1(self):
        """Method to check in-data shape"""
        
        columns_names = self.data.columns
        dim = len(columns_names)
        
        X_columns_names = columns_names[:dim-1]
        y_column_name = columns_names[-1]

        self.X_array = np.array(self.data[X_columns_names])
        self.y_vector = np.array(self.data[y_column_name])

        X_shape = self.X_array.shape
        y_shape = self.y_vector.shape

        assert X_shape[1] < 10, "More then allowed predictors !"
        assert y_shape[0] < 10000, "More then allowed records !"
        print("In data shape is accurate")
            
















