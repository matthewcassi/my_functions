import pandas as pd
import numpy as np

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