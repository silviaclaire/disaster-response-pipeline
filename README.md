# Disaster Response Pipeline Project

A web app implementing a machine learning pipeline to categorize real messages that were sent during disaster events.

### Dependencies
- python 3.7
- nltk
- flask
- joblib
- plotly
- pandas
- sqlalchemy
- scikit-learn

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves the model

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files Explanation

- data
  - process_data.py
    - run ETL pipeline that cleans and stores data in database
- models
  - train_classifier.py
    - run ML pipeline that trains classifier and saves the model
- app
  - run.py
    - run web app after getting the database and the classidier model
