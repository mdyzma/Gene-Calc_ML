from Ml_model_class import ML_model
from sklearn.model_selection import GridSearchCV

class Model_constructor(ML_model):
    def __init__(path, delimiter_type, model_type, best_model):
        super().__init__(path, delimiter_type, model_type)

    