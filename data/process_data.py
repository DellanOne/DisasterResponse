import re
import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    This founction is to load the original filepath.
    
    Summary or Description of the Function

    Parameters:
    value : messages_filepath and categories_filepath.

    Returns:
    Datefame:Returning the new original file.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on=['id'])
  
    return df

def clean_data(df):
    """
    This founction is to clean the new original file.
    
    Summary or Description of the Function

    Parameters:
    Datefame : the new original file.

    Returns:
    Datefame:Returning the new clean file.


    """
    categories = df['categories'].str.split(';',expand=True)
 
    row = categories.loc[0]

    category_colnames = [ re.sub('-.*$', '', x) for x in row ]
 
    categories.columns = category_colnames

    for column in categories:

        categories[column] = categories[column].str.replace('^.*-', '')

        categories[column] = categories[column].astype(int)


    df.drop(['categories'], axis = 1, inplace = True)


    df = pd.merge(df, categories, left_on = df.index.values, right_on = categories.index.values)
    df.drop(['key_0'], axis = 1, inplace = True)

    df = df.drop_duplicates()


    return df

def save_data(df, database_filename):
    """
    This founction is to safe the clean file to dbfile.
    
    Summary or Description of the Function

    Parameters:
    Datefame : the new clean file.

    NO Returns:


    """

    ## Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index = False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()