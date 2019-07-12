from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier

class Model_constructor():
    
    def __init__(self, best_model, X_train, X_test, y_train, y_test):        
        
        self.selected_model = best_model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def grid_search(self):

        """IN PRESS
        Method to build model using best alghoritms with best hyperparameters 
        [from Pre_model construcor class]
        
        Best models options bellow.
        for classifications:
        Random forest classification, KNN classification, 
        Logistic regression
    
        for regression: 
        Simple linear regression, Random forest regression,
        Lasso linear regression, Ridge linear regression
        """
        def rf_classification_gs():
            parameters = {"n_estimators": [50, 500], "bootstrap": ("True", "False")}

            rfc = RandomForestClassifier(random_state=101)
            gs_rfc = GridSearchCV(rfc, parameters, cv=5)
            gs_rfc.fit(self.X_train, self.y_train)
            hyperparameters_res = gs_rfc.best_params_
            accuracy_gs = gs_rfc.best_score_

            print(hyperparameters_res, accuracy_gs)

        def knn_classification_gs():

            k_values = []
            for values in range(30):
                if values % 2 != 0:
                    k_values.append(values)

            parameters = {"n_neighbors": k_values, "weights": ("uniform", "distance"), 
            "algorithm": ("ball_tree", "kd_tree", "brute"), "p": [1, 2]}
            
            knn = KNeighborsClassifier()
            gs_knn = GridSearchCV(knn, parameters, cv=5)
            gs_knn.fit(self.X_train, self.y_train)

            hyperparameters_res = gs_knn.best_params_
            accuracy_gs = gs_knn.best_score_

            print(hyperparameters_res, accuracy_gs)

        def lr_classification_gs():
            parameters = {"solver": ("newton-cg", "lbfgs", "sag", "saga", "liblinear")}

            lr = LogisticRegression(multi_class='auto')
            gs_lr = GridSearchCV(lr, parameters, cv=5)
            gs_lr.fit(self.X_train, self.y_train)

            hyperparameters_res = gs_lr.best_params_
            accuracy_gs = gs_lr.best_score_

            print(hyperparameters_res, accuracy_gs)

        lr_classification_gs()

        def sl_regression_gs():
            parameters = {}
            pass

        def rf_regression_gs():
            parameters = {}
            pass

        def lss_regression_gs():
            parameters = {}
            pass

        def rg_regression_gs():
            parameters = {}
            pass