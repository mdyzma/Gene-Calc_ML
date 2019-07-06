from Source.Ml_model_class import ML_model

models = ["classification", "regression"]
sets_paths = ["Data_set/iris.csv", "Data_set/USA_Housing.csv"]

if __name__ == "__main__":

    model_creator = ML_model(path=sets_paths[0], delimiter_type=",", 
    model_type=models[0])

    model_creator.load_data()
    model_creator.model_selection()
