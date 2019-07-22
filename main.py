from source.pre_model_constructor import Pre_model_constructor
from source.model_constructor import Model_constructor
from source.models_collection import Models
#from source.validation import Validation

#NOTE Type of model: regression or classification is selected by user
#NOTE also data set used to training models.
#NOTE Bellow is just a temporary solution for tests needs

models = ["classification", "regression"]
sets_paths = ["data_sets/iris.csv", "data_sets/USA_Housing.csv"]
val = 1

if __name__ == "__main__":

    pre_model_creator = Pre_model_constructor(path=sets_paths[val], delimiter_type=",",
                                              model_type=models[val])

    input_data = pre_model_creator.load_data()
    data_dict = pre_model_creator.data_set_split(data=input_data) #normalization deafult = False
    
    X_columns = data_dict.get("X_array")
    y_vector = data_dict.get("y_vector")

    X_train = data_dict.get("X_train")
    X_test = data_dict.get("X_test")
    y_train = data_dict.get("y_train")
    y_test = data_dict.get("y_test")

    #TODO data input validation is needed
    
    pre_model_creator.best_model_selection() #method return dict with accuracy scores for evry model
    
    #NOTE temporary solution, best model is selected by cross validation
    best_model = pre_model_creator.models_selector() 

    model_creator = Model_constructor(best_model, X_train, y_train)
    hyperparameters, gs_accuracy = model_creator.grid_search()
    
    model_ready = Models(X_train, X_test, y_train, y_test)
    
    if val == 0: #classifications models

        if best_model == "Random forest classification":
            model, predicted = model_ready.rf_classification(**hyperparameters)

        elif best_model == "KNN classification":
            model, predicted = model_ready.knn_classification(**hyperparameters)

        elif best_model == "Logistic regression":
            model, predicted = model_ready.lr_classification(**hyperparameters)

    elif val == 1: #regression models

        if best_model == "Simple linear regression":
            model, predicted = model_ready.linear_regression()

        elif best_model == "lasso_regression":
            model, predicted = model_ready.lasso_regression(**hyperparameters)
        
        elif best_model == "Ridge linear regression":
            model, predicted = model_ready.ridge_regression(**hyperparameters)

        elif best_model == "random_forest_regression":
            model, predicted = model_ready.random_forest_regression(**hyperparameters)

    model_ready.accuracy_test(gs_accuracy, predicted, val)
    model_ready.export_model(model, best_model)
    print("Predictors", data_dict.get(X_names))
    model_ready.predict(best_model, data_dict.get("y_name"))
