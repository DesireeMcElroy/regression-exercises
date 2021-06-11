import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_variable_pairs(df):
    sns.pairplot(df, corner = True, kind = 'reg', plot_kws={'line_kws':{'color':'red'}})






def months_to_years(df):
    '''
    This function takes in a dataframe and changes the tenure column from months
    to years by dividing it by 12. It also auto changes to int data type.
    '''
    df['tenure_years'] = df.tenure // 12
    return df




def plot_categorical_and_continuous_vars(df, cols, cats):
    for col in df[cols]:
        sns.relplot(data = df, x = df[col], y = cats, kind = 'scatter')
        plt.show()

    for col in df[cols]:
        sns.jointplot(data = df, x =df[col], y = cats, kind = 'scatter')
        plt.show()
        
    sns.heatmap(df.corr(), cmap ='RdBu', center = 0, annot = True, annot_kws={"size": 15})
    plt.show()