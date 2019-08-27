import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load datasets from csv files.

    Args:
        messages_filepath(string): messages csv file path
        categories_filepath(string): categories csv file path

    Returns:
        df(pd.DataFrame): merged dataset
    """
    # read dataset csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets using the common id
    df = pd.merge(messages, categories, how='inner', on='id')
    return df


def clean_data(df):
    """Clean data.

    Args:
        df(pd.DataFrame): dataset to be cleaned

    Returns:
        df(pd.DataFrame): cleaned dataset
    """
    # create a dataframe of individual category columns
    categories = df.categories.str.split(';', expand=True)

    # use the first row to rename column names for the categories data
    row = categories.iloc[0].to_list()
    categories.columns = [s.split('-')[0] for s in row]

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replace df.categories column with new category columns
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Save dataset in a database.

    Args:
        df(pd.DataFrame): dataset to be saved
        database_filename(string): dataset file name

    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    """Main process."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
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