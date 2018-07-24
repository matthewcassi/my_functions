import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def object_bar_plots(df):
    '''
    Function takes a data frame
    Dataframe is subsetted by taking only the columns with a dtype of 'object'
    Each column is then used to create a bar plot of the value counts
    '''
    objects = df.select_dtypes(include='object')
    cols = objects.columns
    for i in cols:
        pd.value_counts(df[i].dropna()).plot.bar()
        plt.title('Value Counts for ' + i)
        plt.ylabel('Count')
        plt.show()
        print('-----------------------------------------------')

def histogram(df, column, log=False):
    '''
    Function takes a data frame, a column, and whether you want the log transform
    The function creates a histogram for the specified column
    '''
    if log == True:
        sns.distplot(df[column].dropna().apply(np.log))
        plt.title('Histogram and Distribution of ' + column)
        plt.xlabel('Log of ' + column)
    else:
        sns.distplot(df[column].dropna())
        plt.title('Histogram and Distribution of ' + column)
    plt.show()

def boxplot(df, column, log=False):
    '''
    Function takes a data frame, a column, and whether you want the log transform
    The function creates a boxplot for the specified column
    '''
    if log == True:
        sns.boxplot(df[column].dropna().apply(np.log), orient='v')
        plt.title('Box Plot of Log ' + column)
        plt.xlabel('Log of ' + column)
    else:
        sns.boxplot(df[column].dropna(), orient='v')
        plt.title('Box Plot of ' + column)
    plt.show()

def scatterplot(df, x_var, y_var, log=False):
    plt.figure(figsize=(15,9))
    if sum(df[y_var].notnull()) == sum(df[x_var].notnull()):
        if log == True:
            plt.scatter(x=df[x_var].dropna(), y=df[y_var].dropna().apply(np.log), alpha=0.3) 
            plt.title('Scatter Plot of ' + x_var + ' and Log Tranform of ' + y_var)
            plt.xlabel(x_var)
            plt.ylabel('Log Transform of ' + y_var)
        else:
            plt.scatter(x=df[x_var].dropna(), y=df[y_var].dropna(), alpha=0.3)
            plt.title('Scatter Plot of ' + x_var + ' and ' + y_var)
            plt.xlabel(x_var)
            plt.ylabel(y_var)
    else:
        drops = df[[x_var,y_var]].dropna()
        if log == True:
            plt.scatter(x=drops[x_var].dropna(), y=drops[y_var].dropna().apply(np.log), alpha=0.3)
            plt.title('Scatter Plot of ' + x_var + ' and Log Tranform of ' + y_var)
            plt.xlabel(x_var)
            plt.ylabel('Log Transform of ' + y_var)
        else:
            plt.scatter(x=drops[x_var].dropna(), y=drops[y_var].dropna(), alpha=0.3)
            plt.title('Scatter Plot of ' + x_var + ' and ' + y_var)
            plt.xlabel(x_var)
            plt.ylabel(y_var)
    plt.show()
    
def heatmap(df):
    '''
    Function takes in a dataframe and creates a heatmap for all of the numerical features in the dataframe
    '''
    corr = df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(25, 12))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    plt.show()
    
def cat_barplot(df, x_col, y_col, log=False):
    plt.figure(figsize=(17,9))
    if log == True:
        sns.barplot(y=df[x_col], x=df[y_col].dropna().apply(np.log),orient='h',ci=None,palette='Blues_d')
        plt.title('Bar Plot of ' + x_col + ' and Log Tranform of ' + y_col)
        plt.xlabel()
        plt.ylabel('Log Transform of ' + y_col)
    else:
        sns.barplot(y=df[x_col].dropna(), x=df[y_col].dropna(),orient='h',ci=None,palette='Blues_d')
        plt.title('Bar Plot of ' + x_col + ' and ' + y_col)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    plt.show()

def cat_boxplot(df, x_col, y_col, log=False):
    plt.figure(figsize=(17,9))
    if log == True:
        sns.boxplot(x=df[x_col], y=df[y_col].dropna().apply(np.log),orient='v')
        plt.title('Bar Plot of ' + x_col + ' and Log Tranform of ' + y_col)
        plt.xlabel(x_col)
        plt.ylabel('Log Transform of ' + y_col)
    else:
        sns.boxplot(x=df[x_col].dropna(), y=df[y_col].dropna(),orient='v')
        plt.title('Bar Plot of ' + x_col + ' and ' + y_col)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    plt.show()