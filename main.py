from Source.Pre_model_constructor import Pre_model_constructor
from Source.Model_constructor import Model_constructor
from Source.Models import Models

#NOTE Type of model: regression or classification is selected by user
#NOTE also data set used to training models.
#NOTE Bellow is just a temporary solution for tests needs

models = ["classification", "regression"]
sets_paths = ["Data_set/iris.csv", "Data_set/USA_Housing.csv"]
val = 0

if __name__ == "__main__":

    pre_model_creator = Pre_model_constructor(path=sets_paths[val], delimiter_type=",",
    model_type=models[val])
    #TODO data input validation is needed

    X_columns, y_column, X_train, X_test, y_train, y_test = pre_model_creator.load_data()
    pre_model_creator.model_selection()
    
    #NOTE temporary solution, best model is selected by cross validation
    best_model = pre_model_creator.models_selector() 

    model_creator = Model_constructor(best_model, X_train, X_test, y_train, y_test)
    hyperparameters, accuracy = model_creator.grid_search()
    
    print(hyperparameters)

    model_ready = Models(X_train, X_test, y_train, y_test)
    
    if val == 0: #classifications models

        if best_model == "Random forest classification":
            model = model_ready.rf_classification(**hyperparameters)[0]

        elif best_model == "KNN classification":
            model = model_ready.knn_classification(**hyperparameters)[0]

        elif best_model == "Logistic regression":
            model = model_ready.lr_classification(**hyperparameters)[0]

    elif val == 1: #regression models

        if best_model == "Simple linear regression":
            model = model_ready.linear_regression()

        elif best_model == "lasso_regression":
            model = model_ready.lasso_regression(**hyperparameters)[0]
        
        elif best_model == "Ridge linear regression":
            model = model_ready.ridge_regression(**hyperparameters)[0]

        elif best_model == "random_forest_regression":
            model = model_ready.random_forest_regression(**hyperparameters)[0]

    print(X_columns)
    model_ready.predict(model, y_column)

    #NOTE in this step user need to select the best model (from trained collection); based on: accuracy, 
    #NOTE cross validation and other metrics
    
    #NOTE only selected model will be proceed in next step (grid search)