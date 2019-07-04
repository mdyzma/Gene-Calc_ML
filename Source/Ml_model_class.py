import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


class ML_model():
    
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        

    def load_data(self):
       
        """
        Load data in csv, xls or xlsx format as data set for model training 
        and evaluation
        """
        
        def data_type(path):

            if path.endswith(".csv"):
                return "csv"
            elif path.endswith(".xlsx") or path.endswith(".xls"):
                return "xlsx"

        def data_set_split(data):
            """
            User need to determine what are X varaibles and y in input data set
            bellow is just temporary solution
            """
            X = np.array(data[["sepal_length", "sepal_width", "petal_length",
            "petal_width"]])
            
            y = np.array(data[["species"]]).ravel()

            return(X, y)

        suffix = data_type(path=self.data_path)
    
        if suffix == "csv":
            data_set = pd.read_csv(self.data_path, delimiter=self.delimiter_type)
       
        elif suffix == "xlsx" or suffix == "xls":
            data_set = pd.read_excel(self.data_path)

        X, y = data_set_split(data_set)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.30, random_state=101)

    def model_selection(self):
        """
        Method to select accurate model for regression or calssification problem based 
        off accuracy and cross validation.

        Regression models: Random Forest and Linear Regression
        Clasfication models: KNN, Random Forest, Logistic Regression
        """

        def primary_model_evaluation(model, Y_true, Y_predicted):
            """method to basic evaluation of possible models"""

            accuracy = accuracy_score(Y_true, Y_predicted)
            cross_validate_score = cross_validate(model, self.X_train, 
            self.y_train, cv=5)

            if self.model_type == "classification":
                matrix_report = classification_report(Y_true, Y_predicted)

                model_evaluation_metrics = {
                    "matrix_report": matrix_report,
                    "cross validate score": cross_validate_score,
                    "accuracy": accuracy
                    }

            elif self.model_type == "regression":
                
                model_evaluation_metrics = {
                    "cross validate score": cross_validate_score,
                    "accuracy": accuracy
                    }

            print(model_evaluation_metrics)

        def rf_classification():
            """Random Rorest Classifier"""
            self.rfc = RandomForestClassifier(n_estimators=100, random_state=101)
            self.rfc.fit(self.X_train, self.y_train)
            predicted = self.rfc.predict(self.X_test)
            
            return(predicted)
        
        def knn_classification():
            """K Neighbors Classifier"""
            dict_of_results = {} # for k-n model

            for k in range(1, 20):
                
                if k % 2 != 0:
                    self.knn = KNeighborsClassifier(n_neighbors=k)
                    self.knn.fit(self.X_train, self.y_train)
                    predicted = self.knn.predict(self.X_test)
                    dict_of_results.update({k: predicted})
            
            return(predicted)
        
        def lr_classification():
            """Logistic Regression Classifier
            #type of solver is just temporary, to consider more flexible options#
            """

            self.lr = LogisticRegression(random_state=101, solver='newton-cg', multi_class='auto')
            self.lr.fit(self.X_train, self.y_train)
            predicted = self.lr.predict(self.X_test)
            
            return(predicted)


        if self.model_type == "classification":
            predicted = rf_classification()
            primary_model_evaluation(self.rfc, self.y_test, predicted)
            
            predicted = knn_classification()
            primary_model_evaluation(self.knn, self.y_test, predicted)

            predicted = lr_classification()
            primary_model_evaluation(self.lr, self.y_test, predicted)
        
        elif self.model_type == "regression" :
            pass

