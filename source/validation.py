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

    def data_NaN(self):
        """Method to find NaN"""

        if np.isnan(self.X_array).any() == True:
            print("NaN in X_array data !")
            exit()
        
        if self.model_type == "regression":
            if np.isnan(self.y_vector).any() == True:
                print("NaN in y_vector data !")
                exit()

    def data_quality(self):
        """Method to find string in input data. In case of X_array strings are not allowed,
        in case of categrocial models in y_vector strings are allowed"""

        try:
            self.X_array.astype(float)
        
        except ValueError:
            print("Data in X_array must be numerical")
            print("If data is categorical convert alphabetic labels to corresponding numbers")
            exit()

        if self.model_type == "regression":        
            try:
                self.y_vector.astype(float)
            
            except ValueError:
                print("Data in y_vector must be numerical")
                exit()

    def extrapolation_risk(X_array, input_predict, X_names):
        n_columns = X_array.shape[1]
        
        try:
            input_predict = np.reshape(input_predict, (n_columns, 1))
        
        except ValueError:
            print("Number of inputs different from number of predictors!")
            exit()

        std_list = np.std(X_array, axis=0)
        mean_list = np.mean(X_array, axis=0)

        for counter, in_values in enumerate(input_predict):
            if in_values < (mean_list[counter] - (3 * (std_list[counter]))):
                print("Risk of extrapolation {} value is smaller than 3 std from mean!". format(X_names[counter]))
            
            elif in_values > (mean_list[counter] + (3 * (std_list[counter]))):
                print("Risk of extrapolation {} value is bigger than 3 std from mean!". format(X_names[counter]))
    
            
        




        

            
        
        
         