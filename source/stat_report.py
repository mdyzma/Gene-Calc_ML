import numpy as np
import scipy.stats as stats
import itertools
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import os

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
        
        try:
            os.mkdir("{}/{}".format(self.stat_report_path, self.project_name))
        except FileExistsError:
            pass

        #TODO add function for xls/xlsx files

    def desc_stat(self):
        """Method to get basic statistic"""
        desc_data = self.data.describe()
        buf = "{}/{}/Statisitical_report_{}.html".format(self.stat_report_path, self.project_name, self.model_type)
        desc_data.to_html(buf)

    def plot(self):
        """Method to generate relation-plot between all vs all in data"""

        if self.model_type == "regression":
            
            data_combinations = list(itertools.combinations(self.col_names, 2))

            for var_x, var_y in data_combinations:
                x_ax = self.data[var_x]
                y_ax = self.data[var_y]
                
                sns.set(style="darkgrid")
                plot = sns.jointplot(x=x_ax, y=y_ax, kind='reg').annotate(stats.pearsonr)
                plot.savefig("{}/{}/{}_vs_{}_{}.png".format(self.stat_report_path, self.project_name, var_x, var_y, self.model_type))

        elif self.model_type == "classification":
            #Generate violin-plot y_var vs all

            quantitive_loc = self.col_names[-1] #y_var
            self.col_names.remove(quantitive_loc)

            for predictor in self.col_names:
                plot = sns.violinplot(data=self.data, x=quantitive_loc, y=predictor)
                fig = plot.get_figure()
                fig.savefig("{}/{}/{}_vs_{}_{}.png".format(self.stat_report_path, self.project_name, quantitive_loc, predictor, self.model_type))
                fig.clear()
            #Generate plot all predictors vs all

            data_combinations = list(itertools.combinations(self.col_names, 2))

            for var_x, var_y in data_combinations:
                x_ax = self.data[var_x]
                y_ax = self.data[var_y]
                
                sns.set(style="darkgrid")
                plot = sns.jointplot(x=x_ax, y=y_ax, kind='reg').annotate(stats.pearsonr)
                plot.savefig("{}/{}/{}_vs_{}_{}.png".format(self.stat_report_path, self.project_name, var_x, var_y, self.model_type))

