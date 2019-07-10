from Ml_model_class import Pre_model_constructor
from sklearn.model_selection import GridSearchCV

class Model_constructor(Pre_model_constructor):
    def __init__(self, path, delimiter_type, model_type, best_model):
        super().__init__(path, delimiter_type, model_type)

    