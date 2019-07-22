from source.pre_model_constructor import Pre_model_constructor
import pytest

models = ["classification", "regression"]
sets_paths = ["data_sets/Iris.csv", "data_sets/USA_Housing.csv"]


def test_instance_data_loading():
    "test function to check instance creation, data loading and sets splitting"
    
    for counter, mod in enumerate(models):
        
        pre_model_creator = Pre_model_constructor(path=sets_paths[counter], delimiter_type=",", 
                                      model_type=mod)

        input_data = pre_model_creator.load_data()
        data_dict = pre_model_creator.data_set_split(data=input_data)

        X_train = data_dict.get("X_train")
        y_test = data_dict.get("y_test")

        if mod == "classification":

            assert X_train.shape == (105, 4), "Data loading or pre-preparing failed"
            assert y_test.shape == (45,), "Data loading or pre-preparing failed"
            
        elif mod == "regression":
            
            assert X_train.shape == (3500, 5), "Data loading or pre-preparing failed"
            assert y_test.shape == (1500,), "Data loading or pre-preparing failed"
    
def test_models():
    """test function to check models trainings"""

    for counter, mod in enumerate(models):
        
        test_model = Pre_model_constructor(path=sets_paths[counter], delimiter_type=",", 
                                      model_type=mod)
        input_data = test_model.load_data()
        test_model.data_set_split(data=input_data)

        if mod == "classification":
            accuracy_dict = test_model.best_model_selection()
            assert accuracy_dict.get("Random forest classification") == 0.9332467532467532,"Random forest classification model prediction failed"
            assert accuracy_dict.get("KNN") == 0.9232467532467533,"KNN classification model prediction failed"
            assert accuracy_dict.get("Logistic regression") == 0.9609090909090909,"Logistic classification model prediction failed"
        
        elif mod == "regression":
            accuracy_dict = test_model.best_model_selection()
            assert accuracy_dict.get("Simple linear regression") == 0.9173373021738389,"Simple linear regression model prediction failed"
            assert accuracy_dict.get("Random forest regression") == 0.8801737815707658,"Random forest regression model prediction failed"
            assert accuracy_dict.get("Lasso linear regression") == 0.9173373065060977,"Lasso regression model prediction failed"
            assert accuracy_dict.get("Ridge linear regression") == 0.9173375618293227,"Ridge regression model prediction failed"