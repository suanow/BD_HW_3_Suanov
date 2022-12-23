from connector.data import get_data
from conf.conf import logging, settings
from util.util import save_model, load_model

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


settings.load_file()


def split(df) -> pd.DataFrame:
    """
    Splitting the dataframe into train and test dataframes
    """
    
    logging.info('Defining X and y')
    
    # splitting on X and Y
    X = df.iloc[:, :-1]
    y = df['target']
    
    logging.info('Splitting on train and test')
    
    # splitting on trains and tests
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=1337)
    
    return X_train, X_test, y_train, y_test


def grid_ranf(X_train, y_train) -> dict:
    """
    Finding the optimal hyperparameters for a RandomForestClassifier model using grid search
    """
    model = RandomForestClassifier()
    
    param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 7, 10]}
    
    grid_search = GridSearchCV(model, param_grid, cv=5)    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_


def grid_svc(X_train, y_train) -> dict:
    """
    Find the optimal hyperparameters for an SVC model using grid search
    """
    model = SVC()
    
    param_grid = {
    'C': [0.1, 1, 5, 10],
    'gamma': [0.1, 1, 10]}

    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_



def training(ModelClass, X_train:pd.DataFrame, y_train:pd.DataFrame, **kwargs) -> any:
    """ 
    Training the choosen model
    """
    # initialize the model
    model = ModelClass(**kwargs)
    
    # train the model
    logging.info(f'Training the {ModelClass.__name__} model')
    model.fit(X_train, y_train)
    
    # saving model to pickle
    save_model(f'model/conf/{ModelClass.__name__}.pkl', model)
    
    # printing accuracy score
    accuracy_score = round(model.score(X_train, y_train) * 100, 2)
    logging.info(f'{ModelClass.__name__} accuracy is {accuracy_score}')
    
    return model


def predict(values, model_path) -> list:
    """
    Making a prediction based on the input values and the model we choose
    """
    
    # reshaping the list
    values = np.array(values).reshape(1, -1)
    model = load_model(model_path)
    logging.info('Model loaded')
    
    return model.predict(values)


df = get_data(settings.DATA.data_set)
X_train, X_test, y_train, y_test = split(df)

SVC_params = grid_svc(X_train, y_train)
ranf_params = grid_ranf(X_train, y_train)

training(RandomForestClassifier, X_train, y_train, **ranf_params)
training(SVC, X_train, y_train, **SVC_params)


# python entrypoint.py --values 1 2 3 4 5 6 7 8 9 10 11 12 13 --model_path model/conf/RandomForestClassifier.pkl