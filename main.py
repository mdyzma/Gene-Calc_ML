from Source.Ml_model_class import ML_model

if __name__ == "__main__":
    
    model_type = "classification" # or regression

    model_creator = ML_model(path="data_set/iris.csv", delimiter_type=",", 
    model_type=model_type)

    model_creator.load_data()
    model_creator.model_selection()
