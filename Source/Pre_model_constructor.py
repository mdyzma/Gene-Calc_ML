import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from .models_collection import Models

class Pre_model_constructor():    
    """
    class with methods to: load and check data, select regression or classyfication model
    based off accuracy.

    path = data set file path with excel (xls or xlsx) or csv extension
    delimiter type = for instance ';' or ',' in case of csv files.
    model type = classification or regression
    """

    def __init__(self, path, delimiter_type, model_type):
        
        self.data_path = path
        self.delimiter_type = delimiter_type
        self.model_type = model_type
        self.models_accuracy = {}
        
    def load_data(self):
        """
        Load data in csv, xls or xlsx format as data set for model training 
        and evaluation
        """
        
        def data_type(data):
            if data.endswith(".csv"):
                return "csv"

            elif data.endswith(".xlsx") or data.endswith(".xls"):
                return "xlsx"

        suffix = data_type(self.data_path)

        if suffix == "csv":
            data_set = pd.read_csv(self.data_path, delimiter=self.delimiter_type)
       
        elif suffix == "xlsx" or suffix == "xls":
            data_set = pd.read_excel(self.data_path)
        
        col_names = data_set.columns
        dim = len(col_names)
        X_columns = col_names[:dim-1]
        y_column = col_names[-1]
        
        X_array = np.array(data_set[X_columns])
        y_vector = np.array(data_set[y_column]).ravel()
        
        data_in = {"X_array": X_array, "y_vector": y_vector, 
                 "X_names": X_columns, "y_name": y_column}

        return data_in

    def data_set_split(self, X, y, normalization=False):
            """
            User need to determine what are X varaibles and y in input data set
            bellow is just temporary.
            Temporary solution is that last column in data set is always y-variable
            return dict:{"X_train": self.X_train, "X_test": self.X_test, 
            "y_train": self.y_train, "y_test": self.y_test}
            """

            #NOTE if normalization == True input data is normalized, optional
            
            if normalization == True: 
                scaler = StandardScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                print("Standard scaler turned on")
            
            elif normalization == False:
                print("Standard scaler turned off")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.30, random_state=101)

            data_dict = {"X_train": self.X_train, "X_test": self.X_test, 
                         "y_train": self.y_train, "y_test": self.y_test,
                         }
            
            return data_dict


    def best_model_selection(self):
        """
        Method to select accurate model for regression or calssification problem based 
        off accuracy and cross validation.
        Regression models: Random Forest and Linear Regression
        Clasfication models: KNN, Random Forest, Logistic Regression
        """

        def primary_model_evaluation(model, model_name, Y_true, Y_predicted):
            """method to basic evaluation of models collection
            model = instance of model
            model_name = name of algorithm
            Y_true = y_test values
            Y_predicted = y_values predicted by mentioned model 
            """
            cross_validate_score = np.mean(cross_validate(model, self.X_train, 
                                           self.y_train, cv=5)["test_score"])

            if self.model_type == "classification":
                accuracy = accuracy_score(Y_true, Y_predicted)
                matrix_report = classification_report(Y_true, Y_predicted, output_dict=True)
                
                df_matrix_report = pd.DataFrame(matrix_report)
                
                model_evaluation_metrics = {
                    "matrix_report": df_matrix_report,
                    "cross validate score": cross_validate_score,
                    "accuracy": accuracy
                    }

                print("{}, evaluation: {}".format(model_name, 
                model_evaluation_metrics))

            elif self.model_type == "regression":
                mae = mean_absolute_error(Y_true, Y_predicted)
                mse = mean_squared_error(Y_true, Y_predicted)
                r2 = r2_score(Y_true, Y_predicted)

                model_evaluation_metrics = {
                    "cross validate score": cross_validate_score,
                    "MAE": mae,
                    "MSE": mse,
                    "R2": r2
                    }

                print("{}, evaluation: {}".format(model_name, 
                model_evaluation_metrics))
            
            return model_evaluation_metrics

        models = Models(self.X_train, self.X_test, self.y_train, self.y_test)

        if self.model_type == "classification":

            rfc, predicted_rf = models.rf_classification()
            rf_model_evaluation_metrics = primary_model_evaluation(rfc, 
            "Random forest classification", self.y_test, predicted_rf)

            knn, predicted_knn = models.knn_classification()
            knn_model_evaluation_metrics = primary_model_evaluation(knn, 
            "KNN classification", self.y_test, predicted_knn)

            lr, predicted_lr = models.lr_classification()
            lr_model_evaluation_metrics = primary_model_evaluation(lr, 
            "Logistic regression classification", self.y_test, predicted_lr)

            svm, predicted_svm = models.svm_classification()
            svm_model_evaluation_metrics = primary_model_evaluation(svm, 
            "Supported vector machines classification", self.y_test, predicted_svm)
            
            self.models_accuracy.update({"Random forest classification": rf_model_evaluation_metrics.get("cross validate score"), 
            "KNN": knn_model_evaluation_metrics.get("cross validate score"), 
            "Logistic regression": lr_model_evaluation_metrics.get("cross validate score"),
            "Supported vector machines classification": svm_model_evaluation_metrics.get("cross validate score")
            })
                    
        elif self.model_type == "regression":

            lreg, predicted_linear = models.linear_regression()
            llinear_model_evaluation_metrics = primary_model_evaluation(lreg, 
            "Simple linear regression", self.y_test, predicted_linear)

            rfr, predicted_rf = models.random_forest_regression()
            rf_model_evaluation_metrics = primary_model_evaluation(rfr, 
            "Random forest regression", self.y_test, predicted_rf)
            
            lasso, predicted_lasso = models.lasso_regression()
            lasso_model_evaluation_metrics = primary_model_evaluation(lasso, 
            "Lasso linear regression", self.y_test, predicted_lasso)
            
            ridge, predicted_ridge = models.ridge_regression()
            ridge_model_evaluation_metrics = primary_model_evaluation(ridge, 
            "Ridge linear regression", self.y_test, predicted_ridge)


            self.models_accuracy.update({"Simple linear regression": llinear_model_evaluation_metrics.get("cross validate score"), 
            "Random forest regression": rf_model_evaluation_metrics.get("cross validate score"), 
            "Lasso linear regression": lasso_model_evaluation_metrics.get("cross validate score"),
            "Ridge linear regression": ridge_model_evaluation_metrics.get("cross validate score")
            })
        
        return self.models_accuracy #for test needs
        
    def models_selector(self):

        """method to type best model for current problem from trained collection
        input is a dict with models accuracy scores [cross validation score]
        output is a model name with the best mean accuracy
        """

        if self.model_type == "classification":
            best_model = max(self.models_accuracy, key=self.models_accuracy.get)
            
        elif self.model_type == "regression":
            best_model = max(self.models_accuracy, key=self.models_accuracy.get)

        print("Propably best model is: {}".format(best_model))
        
        return best_model

        