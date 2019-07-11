from Pre_model_constructor import Pre_model_constructor
from .Models import Models
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

class Model_constructor(Pre_model_constructor):
    
    def __init__(self, path, delimiter_type, model_type, best_model):
        super().__init__(path, delimiter_type, model_type)
        
        self.selected_model = best_model

    def boost_models(self):
        """
        Method to build model using best alghoritms 
        [from Pre_model construcor class]
        method return trained model for grid search
        
        Best models options bellow.
        for classifications:
        Random forest classification, KNN classification, 
        Logistic regression
    
        for regression: 
        Simple linear regression, Random forest regression,
        Lasso linear regression, Ridge linear regression
        """

      

    