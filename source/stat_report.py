import numpy as np
import scipy.stats as stats
import itertools
import seaborn as sns 
import pandas as pd

class Report_generator():

    def __init__(self, stat_report, data_set, model_type, alghoritm, project_name):
        self.stat_report_path = stat_report
        self.data_set = data_set
        self.model_type = model_type
        self.alghoritm = alghoritm
        self.project_name = project_name
    
    def prepare_data(self):
        self.data = pd.read_csv(self.data_set)
        self.col_names = list(self.data.columns)

        for columns in self.col_names:
            
            try: #delete column if contain string
                np.array(self.data[columns]).astype(float)
            except:
                self.col_names.remove(columns)
                print("Column '{}' is not numerical so it can not be used in plot".format(columns))

        #TODO add function for xls/xlsx files

    def desc_stat(self):
        """Method to get basic statistic"""
        desc_data = self.data.describe()
        buf = "{}/Statisitical_report_{}.html".format(self.stat_report_path, self.project_name)
        desc_data.to_html(buf)

    
    def plot(self):
        """Method to generate relation-plot between all vs all in data"""
        data_combinations = list(itertools.combinations(self.col_names, 2))

        for var_x, var_y in data_combinations:
            x_ax = self.data[var_x]
            y_ax = self.data[var_y]
            
            sns.set(style="darkgrid")
            plot = sns.jointplot(x=x_ax, y=y_ax, kind='reg').annotate(stats.pearsonr)
            plot.savefig("{}/{}_{}_vs_{}.png".format(self.stat_report_path, self.project_name, var_x, var_y))





