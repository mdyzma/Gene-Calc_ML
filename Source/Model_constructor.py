from .Pre_model_constructor import Pre_model_constructor
from .Models import Models
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

class Model_constructor(Models):
    
    def __init__(self, best_model, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        
        self.selected_model = best_model

    def grid_search(self):

        """
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
            pass

        def knn_classification_gs():
            pass

        def lr_classification_gs():
            pass

        def sl_regression_gs():
            pass

        def rf_regression_gs():
            pass

        def lss_regression_gs():
            pass

        def rg_regression_gs():
            pass
      

    