from Source.Pre_model_constructor import Pre_model_constructor
from Source.Model_constructor import Model_constructor
from Source.Models import Models

#NOTE Type of model: regression or classification is selected by user
#NOTE also data set used to training models.
#NOTE Bellow is just a temporary solution for tests needs

models = ["classification", "regression"]
sets_paths = ["Data_set/iris.csv", "Data_set/USA_Housing.csv"]
val = 1

if __name__ == "__main__":

    pre_model_creator = Pre_model_constructor(path=sets_paths[val], delimiter_type=",",
    model_type=models[val])
    #TODO data input validation is needed

    X_train, X_test, y_train, y_test = pre_model_creator.load_data()
    pre_model_creator.model_selection()
    
    #NOTE temporary solution, best model is selected by cross validation
    best_model = pre_model_creator.models_selector() 

    model_creator = Model_constructor(best_model, X_train, X_test, y_train, y_test)
    hyperparameters, accuracy = model_creator.grid_search()
    
    print(hyperparameters)

    model_ready = Models(X_train, X_test, y_train, y_test)
    
    if best_model == "Logistic regression":
        model_ready.lr_classification(**hyperparameters)

    elif best_model == "Ridge linear regression":
        model_ready.ridge_regression(**hyperparameters)

        

    #NOTE in this step user need to select the best model (from trained collection); based on: accuracy, 
    #cross validation and other metrics
    
    #NOTE only selected model will be proceed in next step (grid search)