from Source.Pre_model_constructor import Pre_model_constructor
from Source.Model_constructor import Model_constructor

#NOTE Type of model: regression or classification is selected by user
#NOTE also data set used to training models.
#NOTE Bellow is just a temporary solution for tests needs

models = ["classification", "regression"]
sets_paths = ["Data_set/iris.csv", "Data_set/USA_Housing.csv"]

if __name__ == "__main__":

    pre_model_creator = Pre_model_constructor(path=sets_paths[1], delimiter_type=",",
    model_type=models[1])
    #TODO data input validation is needed

    pre_model_creator.load_data()
    pre_model_creator.model_selection()
    best_model = pre_model_creator.models_selector() # temorary solution, best model is selected by cross validation

    # model_constructor = Model_constructor(path=sets_paths[0], delimiter_type=",",
    # model_type=models[0], best_model="Random forest classification")
    # model_constructor.load_data()

    #NOTE in this step user need to select the best model (from trained collection); based on: accuracy, 
    #cross validation and other metrics
    
    #NOTE only selected model will be proceed in next step (grid search)