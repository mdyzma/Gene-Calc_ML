from source.pre_model_constructor import Pre_model_constructor
from source.model_optimizer import Model_optimizer
import pytest

models = ["classification", "regression"]
sets_paths = ["data_sets/Iris.csv", "data_sets/USA_Housing.csv"]
models_id = {1: "Random forest classification", 2: "KNN classification", 
             3: "Logistic regression", 4: "Supported vector machines classification", 
             5: "Simple linear regression", 6: "Lasso linear regression", 
             7: "Ridge linear regression", 8: "Random forest regression"}

#TODO function to test validation method from Validation class 

def test_instance_data_loading():
    "test function to check instance creation, data loading and sets splitting"
    
    for counter, mod in enumerate(models):
        
        pre_model_creator = Pre_model_constructor(path=sets_paths[counter], delimiter_type=",", 
                                                  model_type=mod, models_id=models_id)

        pre_model_creator.load_data()

        data_dict = pre_model_creator.data_set_split()

        X_train = data_dict.get("X_train")
        y_test = data_dict.get("y_test")

        if mod == "classification":

            assert X_train.shape == (105, 4), "Data loading or pre-preparing failed"
            assert y_test.shape == (45,), "Data loading or pre-preparing failed"
            
        elif mod == "regression":
            
            assert X_train.shape == (3500, 5), "Data loading or pre-preparing failed"
            assert y_test.shape == (1500,), "Data loading or pre-preparing failed"
    
def test_pre_models():
    """test function to check pre_models works"""

    for counter, mod in enumerate(models):
        
        test_model = Pre_model_constructor(path=sets_paths[counter], delimiter_type=",", 
                                           model_type=mod, models_id=models_id)
        test_model.load_data()

        test_model.data_set_split()
        
        if mod == "classification":
            accuracy_dict = test_model.best_model_selection()
            assert accuracy_dict.get(1) == 0.9332467532467532,"Random forest classification model prediction failed"
            assert accuracy_dict.get(2) == 0.9232467532467533,"KNN classification model prediction failed"
            assert accuracy_dict.get(3) == 0.9414285714285715,"Logistic classification model prediction failed"
            assert accuracy_dict.get(4) == 0.9609523809523809,"SVM classification model prediction failed"
        elif mod == "regression":
            accuracy_dict = test_model.best_model_selection()

            assert accuracy_dict.get(5) == 0.9168489817090973,"Simple linear regression model prediction failed"
            assert accuracy_dict.get(6) == 0.9168489860285964,"Lasso regression model prediction failed"
            assert accuracy_dict.get(7) == 0.9168492456702015,"Ridge regression model prediction failed"
            assert accuracy_dict.get(8) == 0.8794821066207236,"Random forest regression model prediction failed"

def test_pre_models_normalize():
    """test function to check pre_models work (models traning on normalized data)"""
    
    for counter, mod in enumerate(models):
        test_model = Pre_model_constructor(path=sets_paths[counter], delimiter_type=",", 
                                            model_type=mod, models_id=models_id)
        test_model.load_data()

        test_model.data_set_split(normalization=True)

        if mod == "classification":
            accuracy_dict = test_model.best_model_selection()
            assert accuracy_dict.get(1) == 0.9332467532467532,"Normalization turned on, Random forest classification model prediction failed"
            assert accuracy_dict.get(2) == 0.9323376623376625,"Normalization turned on, KNN classification model prediction failed"
            assert accuracy_dict.get(3) == 0.9414285714285715,"Normalization turned on, Logistic classification model prediction failed"
            assert accuracy_dict.get(4) == 0.9514285714285714,"SVM classification model prediction failed"

        elif mod == "regression":
            accuracy_dict = test_model.best_model_selection()
            assert accuracy_dict.get(5) == 0.9168489817090959,"Normalization turned on, Simple linear regression model prediction failed"
            assert accuracy_dict.get(6) == 0.9168489847392213,"Normalization turned on, Lasso regression model prediction failed"
            assert accuracy_dict.get(7) == 0.9168494020921008,"Normalization turned on, Ridge regression model prediction failed"
            assert accuracy_dict.get(8) == 0.8794843416507467,"Normalization turned on, Random forest regression model prediction failed"

def test_GridSearch_classification():
    """function to test Grid Search for classification models with 
    raw and normalized data #NOTE VERY SLOW"""
    
    best_models = [1, 2, 3] # Random forest classification", "KNN classification", "Logistic regression"
    results = [0.9332467532467532, 0.9614285714285714, 0.9414285714285715]
    results_normalized = [0.9332467532467532, 0.9514285714285714, 0.9414285714285715]

    test_model = Pre_model_constructor(path="data_sets/Iris.csv", delimiter_type=",", 
                                       model_type="classification", models_id=models_id)

    test_model.load_data()

    data_dict = test_model.data_set_split()
    X_train = data_dict.get("X_train")
    y_train = data_dict.get("y_train")

    for counter, model in enumerate(best_models):
        grid_model = Model_optimizer(model, X_train, y_train, models_id)
        accuracy = grid_model.grid_search()[1]
        assert accuracy == results[counter],"Search grid PROPABLY does not work correctly for classification models"

    data_dict = test_model.data_set_split(normalization=True)
    X_train = data_dict.get("X_train")
    y_train = data_dict.get("y_train")

    for counter, model in enumerate(best_models):
        grid_model = Model_optimizer(model, X_train, y_train, models_id)
        accuracy = grid_model.grid_search()[1]
        assert accuracy == results_normalized[counter],"Normalization turned on, Search grid PROPABLY does not work correctly for classification models"

def test_GridSearch_regression():
    """Function to test Grid Search for regression models with 
    raw and normalized data #NOTE VERY VERY SLOW"""
    
    best_models = [8,6,7] # "Random forest regression", "Lasso linear regression", "Ridge linear regression"
    results = [0.8811994374499792, 0.9168491434818741, 0.9168492237328291]
    normalized_results = [0.8811962679449141, 0.9168491434818741, 0.9168493716508754]

    test_model = Pre_model_constructor(path="data_sets/USA_Housing.csv", delimiter_type=",", 
                                       model_type="classification", models_id=models_id)
    test_model.load_data()

    data_dict = test_model.data_set_split()
    
    X_train = data_dict.get("X_train")
    y_train = data_dict.get("y_train")

    for counter, model in enumerate(best_models):
        grid_model = Model_optimizer(model, X_train, y_train, models_id)
        accuracy = grid_model.grid_search()[1]
        assert accuracy == results[counter],"Search grid PROPABLY does not work correctly for regression models"

    data_dict = test_model.data_set_split(normalization=True)
    
    X_train = data_dict.get("X_train")
    y_train = data_dict.get("y_train")
    
    for counter, model in enumerate(best_models):
        grid_model = Model_optimizer(model, X_train, y_train, models_id)
        accuracy = grid_model.grid_search()[1]
        assert accuracy == normalized_results[counter],"Normalized turned on, Search grid PROPABLY does not work correctly for regression models"
