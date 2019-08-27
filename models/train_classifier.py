import os
import sys
import json
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from joblib import dump
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """Load data from database.

    Args:
        database_filepath(string): dataset file path

    Returns:
        X(pd.Series): message column
        Y(pd.DataFrame): categories columns
        category_names(list): category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.loc[:, 'related':]
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    """Process text data.

    Args:
        text(string): text to be tokenized

    Returns:
        clean_tokens(list): cleaned tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    """Build ML pipeline with GridSearchCV.

    Args:
        None

    Returns:
        model(GridSearchCV): ML pipeline
    """
    # build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier())),
    ], verbose=True)
    # use grid search to find better parameters
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'clf__estimator__n_estimators': (50, 100),
    }
    model = GridSearchCV(pipeline, parameters, cv=3, verbose=1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model with test data and output scores.

    Args:
        model: estimator
        X_test(pd.Series): test data
        Y_test(pd.DataFrame): correct target values for X_test
        category_names(list): category names

    Returns:
        None
    """
    # predict with test data
    Y_pred = model.predict(X_test)

    # output classification report for each category
    for i, cat in enumerate(category_names):
        print(f'category: {cat}')
        print(classification_report(Y_test[cat], Y_pred[:,i], output_dict=False))


def save_model(model, model_filepath):
    """Save model.

    Args:
        model: model to be saved
        model_filepath(string): model file path

    Returns:
        None
    """
    with open(model_filepath, 'wb') as f:
        dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()