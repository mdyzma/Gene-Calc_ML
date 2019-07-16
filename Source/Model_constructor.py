from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from numpy import arange

class Model_constructor():
    
    def __init__(self, best_model, X_train, y_train):        
        
        self.selected_model = best_model
        self.X_train = X_train
        self.y_train = y_train

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
            """Search grid for randof forest classification"""
            parameters = {"n_estimators": [50, 500]}

            rfc = RandomForestClassifier(random_state=101)
            gs_rfc = GridSearchCV(rfc, parameters, cv=5)
            gs_rfc.fit(self.X_train, self.y_train)
            hyperparameters_res = gs_rfc.best_params_
            accuracy_gs = gs_rfc.best_score_

            return(hyperparameters_res, accuracy_gs)

        def knn_classification_gs():
            """Search grid for KNN classification"""

            def odd(x):
                if x % 2 != 0:
                    return x

            k_values = list(filter(odd, range(50)))

            parameters = {"n_neighbors": k_values, "weights": ("uniform", "distance"), 
            "algorithm": ("ball_tree", "kd_tree", "brute"), "p": [1, 2]}
            
            knn = KNeighborsClassifier()
            gs_knn = GridSearchCV(knn, parameters, cv=5)
            gs_knn.fit(self.X_train, self.y_train)

            hyperparameters_res = gs_knn.best_params_
            accuracy_gs = gs_knn.best_score_

            return(hyperparameters_res, accuracy_gs)

        def lr_classification_gs():
            """Search grid for logistic regression classification"""
            
            c_range = list(arange(0.1, 1, 0.1))
            parameters = {"solver": ("newton-cg", "lbfgs", "sag", "saga", "liblinear"), 
            "warm_start": ("True", "False"), "C": c_range}

            lr = LogisticRegression(multi_class='auto')
            gs_lr = GridSearchCV(lr, parameters, cv=5)
            gs_lr.fit(self.X_train, self.y_train)

            hyperparameters_res = gs_lr.best_params_
            accuracy_gs = gs_lr.best_score_

            return(hyperparameters_res, accuracy_gs)

        #NOTE no search gird method for simple linear regression [no needed now]

        def rf_regression_gs():
            """Search grid for random foresr regression"""

            parameters = {"n_estimators": [50, 500]}
            
            rf = RandomForestRegressor()
            gs_rf = GridSearchCV(rf, parameters, cv=5)
            gs_rf.fit(self.X_train, self.y_train)

            hyperparameters_res = gs_rf.best_params_
            accuracy_gs = gs_rf.best_score_

            return(hyperparameters_res, accuracy_gs)

        def lss_regression_gs():
            """Search grid for Lasso regression"""
            
            a_range = list(arange(0.1, 1, 0.1))
            parameters = {"alpha": a_range, "normalize": ("True", "False")}            
            
            lss = Lasso()
            gs_lss = GridSearchCV(lss, parameters, cv=5)
            gs_lss.fit(self.X_train, self.y_train)

            hyperparameters_res = gs_lss.best_params_
            accuracy_gs = gs_lss.best_score_

            return(hyperparameters_res, accuracy_gs)

        def rg_regression_gs():
            """Search grid for Ridge regression"""

            a_range = list(arange(0.1, 1, 0.1))
            parameters = {"alpha": a_range}

            rg = Ridge(solver="auto")
            gs_rg = GridSearchCV(rg, parameters, cv=5)
            gs_rg.fit(self.X_train, self.y_train)

            hyperparameters_res = gs_rg.best_params_
            accuracy_gs = gs_rg.best_score_

            return(hyperparameters_res, accuracy_gs)
        
        def use_best_model():
            hyperparameters, accuracy_gs = ["", ""]
            
            if self.selected_model == "Random forest classification":
                hyperparameters, accuracy_gs = rf_classification_gs()

            elif self.selected_model == "KNN classification":
                hyperparameters, accuracy_gs = knn_classification_gs()

            elif self.selected_model == "Logistic regression":
                hyperparameters, accuracy_gs = lr_classification_gs()

            elif self.selected_model == "Simple linear regression":
                print("Search grid for simple linear regression model no available")

            elif self.selected_model == "Random forest regression":
                hyperparameters, accuracy_gs = rf_regression_gs()

            elif self.selected_model == "Lasso linear regression":
                hyperparameters, accuracy_gs = lss_regression_gs()

            elif self.selected_model == "Ridge linear regression":
               hyperparameters, accuracy_gs = rg_regression_gs()

            return(hyperparameters, accuracy_gs)

        hyperparameters, accuracy_gs = use_best_model()
        return(hyperparameters, accuracy_gs)



            