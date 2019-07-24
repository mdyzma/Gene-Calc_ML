from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

#TODO add naive methods for accuracy comparison

class Models():

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    #NOTE classification models bellow

    def rf_classification(self, n_estimators=100, random_state=101, **kwargs):
        """Random Forest Classifier"""
        rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rfc.fit(self.X_train, self.y_train)
        predicted = rfc.predict(self.X_test)
        
        return(rfc, predicted)
    
    def knn_classification(self, n_neighbors=None, **kwargs):
        """K Neighbors Classifier"""
        """TODO data normalization needed"""

        if n_neighbors == None:
            dict_of_results = {} # for k-n model

            for k in range(1, 10):
                
                if k % 2 != 0:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(self.X_train, self.y_train)
                    predicted = knn.predict(self.X_test)
                    accuracy = accuracy_score(self.y_test, predicted)
                    dict_of_results.update({k: accuracy})
            
            best_k = max(dict_of_results, key=dict_of_results.get)
            knn = KNeighborsClassifier(n_neighbors=best_k)
            knn.fit(self.X_train, self.y_train)
            predicted = knn.predict(self.X_test)
        
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(self.X_train, self.y_train)
            predicted = knn.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predicted)


        return(knn, predicted)
    
    def lr_classification(self, max_iter=50,  C=1, solver="saga", 
                          warm_start=True, multi_class="auto", **kwargs):
        """Logistic Regression Classifier"""

        lr = LogisticRegression(solver=solver, max_iter=max_iter, C=C, 
        warm_start=warm_start, multi_class=multi_class)

        lr.fit(self.X_train, self.y_train)
        predicted = lr.predict(self.X_test)
        
        return(lr, predicted)

    #NOTE regression models bellow

    def linear_regression(self):
        """
        Linear regression model for regression problems
        Ordinary least squares
        """
        lreg = LinearRegression()
        lreg.fit(self.X_train, self.y_train)
        predicted = lreg.predict(self.X_test)

        return(lreg, predicted)

    def lasso_regression(self, alpha=1.0, **kwargs):
        """Lasso regression model"""
        lasso = Lasso(alpha=alpha)
        lasso.fit(self.X_train, self.y_train)
        predicted = lasso.predict(self.X_test)

        return(lasso, predicted)

    def ridge_regression(self, alpha=1.0, **kwargs):
        """Ridge regression model"""
        ridge = Ridge(alpha=alpha)
        ridge.fit(self.X_train, self.y_train)
        predicted = ridge.predict(self.X_test)

        return(ridge, predicted)

    def random_forest_regression(self, random_state=101, n_estimators=100, **kwargs):
        rfr = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators)
        rfr.fit(self.X_train, self.y_train)
        predicted = rfr.predict(self.X_test)

        return(rfr, predicted)

    def export_model(self, model, model_name):
            path = "ready_models/{}".format(model_name)
            joblib.dump(model, path)

    def accuracy_test(self, gs_accuracy, predicted, val):
        """Method return accuracy for test data set [R2 in case of regression models]"""
        if val == 0:
            accuracy = accuracy_score(self.y_test, predicted)
        
        elif val == 1:
            accuracy = r2_score(self.y_test, predicted)

        print("Cross validation [on train set] = {}\nFinall accuracy on test set = {}"
            .format(gs_accuracy, accuracy))

    def predict(self, model_name, y_column_name, normalization=False):
        """method to predict y values using best model with best hyperparameters"""
        path = "ready_models/{}".format(model_name)
        input_values = input("Input X values: separated by commas => ").split(",")
        
        if normalization == False:
            input_values = np.array(input_values).reshape(1, -1).astype(np.float64) #pretyping to float64 needed
        
        elif normalization == True:
            raw_input_values = np.array(input_values).reshape(1, -1).astype(np.float64)
            scaler = StandardScaler()
            scaler.fit(raw_input_values)
            input_values = scaler.transform(raw_input_values)

        model = joblib.load(path)
        print("model loaded")
        predicted_data = model.predict(input_values)
        print("{} = {}".format(y_column_name, predicted_data))