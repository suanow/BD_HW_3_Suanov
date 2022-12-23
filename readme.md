This project contains functions for training and making predictions with machine learning models using the scikit-learn library.

**Functions**

*split(df) -> pd.DataFrame*
This function defines the features X and target y from the dataframe, and then splits the data into train and test sets using train_test_split from scikit-learn. 

*grid_ranf(X_train, y_train) -> dict*
This function performs a grid search to find the optimal hyperparameters for a RandomForestClassifier model. It takes the train set for features X_train and target y_train as inputs and returns a dictionary of the best hyperparameters found.

*grid_svc(X_train, y_train) -> dict*
This function performs a grid search to find the optimal hyperparameters for an SVC model. It takes the train set for features X_train and target y_train as inputs and returns a dictionary of the best hyperparameters found.

*training(ModelClass, X_train, y_train, **kwargs) -> any*
This function trains a given model and saves it to a pickle file. It takes the model class ModelClass, the train set for features X_train and target y_train, and any additional model hyperparameters as keyword arguments **kwargs, returns the trained model.

*predict(values, model_path) -> list*
This function makes a prediction based on a given list of values values and a trained model stored in the file at the given file path model_path, returns a list of predictions.

**How to use**
Type python entrypoint.py --values (your values here) --model_path (model path here)

If you want to change pathes to models or data you can use settings.toml file and apply changes there