from Source.Pre_model_constructor import Pre_model_constructor

models = ["classification", "regression"]
sets_paths = ["Data_set/iris.csv", "Data_set/USA_Housing.csv"]

if __name__ == "__main__":

    model_creator = Pre_model_constructor(path=sets_paths[1], delimiter_type=",",
    model_type=models[1])
    #TODO data input validation is needed

    model_creator.load_data()
    model_creator.model_selection()
    best_model = model_creator.models_selector() # temorary solution, best model is selected by cross validation

    #NOTE in this step user need to select the best model (from trained collection); based on: accuracy, 
    #cross validation and other metrics
    
    #NOTE only selected model will be proceed in next step (grid search)