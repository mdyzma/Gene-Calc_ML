from Source.Ml_model_class import ML_model
import pytest

models = ["classification", "regression"]
sets_paths = ["Data_set/iris.csv", "Data_set/USA_Housing.csv"]


def test_instance_data():
    "test function to check instance creation, data loading and sets splitting"
    
    for counter, mod in enumerate(models):
        
        model = ML_model(sets_paths[counter], ",", mod)

        X_train, y_test = model.load_data()

        if mod == "classification":

            assert X_train.shape == (105, 4), "Data loading or pre-preparing failed"
            assert y_test.shape == (45,), "Data loading or pre-preparing failed"
        
        elif mod == "regression":
            
            assert X_train.shape == (3500, 5), "Data loading or pre-preparing failed"
            assert y_test.shape == (1500,), "Data loading or pre-preparing failed"
    


        
    


