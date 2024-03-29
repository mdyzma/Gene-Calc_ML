from source.pre_model_constructor import Pre_model_constructor
from source.model_optimizer import Model_optimizer
from source.models_collection import Models
from source.validation import Validation
from source.stat_report import Report_generator
import click 

#NOTE Type of model: regression or classification is selected by user
#NOTE also data set used to training models.

def welcome():
    click.echo(click.style('APMC - Automatic Predictive Model Constructor v. 1.0', blink=False, bold=True))
    click.echo(click.style('CLI tool to train and improve machine learning models', fg='white', bold=True))

    click.echo(click.style('\nCreated by Gene-Calc Team -> www.gene-calc.pl', blink=True, bold=True))
    click.echo(click.style('Contact: contact@gene-calc.pl\n', fg='white', bold=True))

def print_help(ctx, param, value):
    if value is False:
        return
    click.echo(ctx.get_help())
    ctx.exit()

@click.command()
@click.option("--data-input", "-i", "data_input", type=click.Path(), help="Path to file with data set [REQUIRED]")

@click.option("--model-type", "-m", "model_type", type=click.Choice(["classification", "regression"], case_sensitive=False), help="Select type of model [REQUIRED]")

@click.option("--project-name", "-p", "project_name", default="Project", type=str,  help="Project name [in case of export model is a file name] [OPTIONAL]")

@click.option("--model-out", "-o", "model_out", type=click.Path(), default=None, help="Path to export trained model for future predictions [OPTIONAL]")

@click.option("--stat-report", "-r", "stat_report", type=click.Path(), default=None, help="Path to export statistical report about input-data, if None report will not rise, deafult is False [OPTIONAL]")

@click.option("--normalization", "-n", "normalization", is_flag=True, default=False, help="Normalize input data, default is False [OPTIONAL]")

@click.option("--delimiter", "-d", type=str, default=",", help=" Select type of delimiter in case of *.csv files, default is ',' [OPTIONAL]")

@click.option("--help", "-h",
 is_flag=True,
 expose_value=False,
 is_eager=False,
 callback=print_help, 
 help="Print help message"
 )

@click.pass_context
def construction_procedure(ctx, data_input, model_type, project_name, model_out, stat_report, normalization, delimiter):
    print("{}\n".format(project_name))

    models_id = {1: "Random forest classification", 2: "KNN classification", 
             3: "Logistic regression", 4: "Supported vector machines classification", 
             5: "Simple linear regression", 6: "Lasso linear regression", 
             7: "Ridge linear regression", 8: "Random forest regression"}


    if (data_input is None) or (model_type is None): #Go to help if no parameters
        print_help(ctx, None,  value=True)

    pre_model_creator = Pre_model_constructor(path=data_input, delimiter_type=delimiter,
                                              model_type=model_type, models_id=models_id)

    data_in = pre_model_creator.load_data()
    X_array = data_in.get("X_array")
    y_vector = data_in.get("y_vector")

    validation = Validation(X=X_array, y=y_vector, model_type=model_type)
    validation.shape_validation()
    validation.data_quality()
    validation.data_NaN()
    
    data_dict = pre_model_creator.data_set_split(normalization=normalization) #normalization deafult = False

    X_train = data_dict.get("X_train")
    X_test = data_dict.get("X_test")
    y_train = data_dict.get("y_train")
    y_test = data_dict.get("y_test")

    pre_model_creator.best_model_selection() #method obtain cross-val accuracy scores for every model
    
    #NOTE best model is selected by cross validation
    pre_model_creator.models_selector()


    if model_type == "classification":
        
        print("""
            1: Random forest classification
            2: KNN classification
            3: Logistic regression
            4: Supported vector machines classification
        """)
    
    elif model_type == "regression":

        print("""
            5: Simple linear regression
            6: Lasso linear regression
            7: Ridge linear regression
            8: Random forest regression
        """)

    if model_type == "classification":
        best_model = click.prompt('Please enter the number of chosen model', type=int)
    elif model_type == "regression":
        best_model = click.prompt('Please enter the number of chosen model', type=int)
    
    click.echo(click.style('Be patient this step may take some time ...', blink=True))
    

    model_creator = Model_optimizer(best_model, X_train, y_train, models_id)
    hyperparameters, gs_accuracy = model_creator.grid_search()
    
    model_ready = Models(X_train, X_test, y_train, y_test)
    
    if model_type == "classification": #classifications models

        if best_model == 1: #"Random forest classification"
            model, predicted = model_ready.rf_classification(**hyperparameters)

        elif best_model == 2: #"KNN classification"
            model, predicted = model_ready.knn_classification(**hyperparameters)

        elif best_model == 3: #"Logistic regression"
            model, predicted = model_ready.lr_classification(**hyperparameters)

        elif best_model == 4: #"Supported vector machines classification"
            model, predicted = model_ready.svm_classification(**hyperparameters)

    elif model_type == "regression": #regression models

        if best_model == 5: # "Simple linear regression"
            model, predicted = model_ready.linear_regression()

        elif best_model == 6: #"Lasso linear regression"
            model, predicted = model_ready.lasso_regression(**hyperparameters)
        
        elif best_model == 7: #"Ridge linear regression"
            model, predicted = model_ready.ridge_regression(**hyperparameters)

        elif best_model == 8: #"Random forest regression"
            model, predicted = model_ready.random_forest_regression(**hyperparameters)


    model_ready.accuracy_test(gs_accuracy, predicted, model_type)
    
    if model_out is not None:
        model_ready.export_model(model, model_out, project_name)

    if stat_report is not None:
        report = Report_generator(best_model, stat_report, data_input, 
                                model_type, model, project_name)
        report.prepare_data()
        report.desc_stat()
        report.get_model_data()
        report.plot()
    
    model_ready.predict(model, models_id.get(best_model), data_in.get("X_names"), 
                        data_in.get("y_name"), 
                        normalization=normalization, 
                        mean_array=data_dict.get("X_mean"),
                        std_array=data_dict.get("X_std")
                        )

if __name__ == "__main__":
    welcome()
    construction_procedure()