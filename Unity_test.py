from Source.Ml_model_class import ML_model
import pytest

models = ["classification", "regression"]
sets_paths = ["Data_set/iris.csv", "Data_set/USA_Housing.csv"]


def test_instance_data_loading():
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
    
def test_classification_methods():
    for counter, mod in enumerate(models):
        
        test_model = ML_model(sets_paths[counter], ",", mod)
        test_model.load_data()

        if mod == "classification":
            rf_val_score, knn_val_score, lr_val_score = test_model.model_selection()
            assert rf_val_score.get("cross validate score") == 0.9332467532467532,"Random forest classification model prediction failed"
            assert knn_val_score.get("cross validate score") == 0.9232467532467533,"KNN classification model prediction failed"
            assert lr_val_score.get("cross validate score") == 0.9414285714285715,"Logistic classification model prediction failed"
        
        elif mod == "regression":
            lr_val_score, rf_val_score, lasso_model, ridge_model = test_model.model_selection()
            assert lr_val_score.get("cross validate score") == 0.9173373021738389,"Simple linear regression model prediction failed"
            assert rf_val_score.get("cross validate score") == 0.8801737815707658,"Random forest regression model prediction failed"
            assert lasso_model.get("cross validate score") == 0.9173373026078024,"Lasso regression model prediction failed"
            assert ridge_model.get("cross validate score") == 0.9173375618293227,"Ridge regression model prediction failed"