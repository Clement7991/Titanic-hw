### IMPORTS ###
import pandas as pd
import os
from google.cloud import storage
from config import Config
from titanic.ml.preprocessing.pipeline import pipeline_train, pipeline_pred
from titanic.ml.model.model_rfc import init_rfc
from sklearn.impute import SimpleImputer


###  ###
def preprocess_data():
    ''' returns preprocessed features and labels for training set'''

    # download data from cloud
    json_keyfile_path = 'titanic/data/titanic-393415-e3b61a50fb0a.json'
    client = storage.Client().from_service_account_json(json_keyfile_path)
    bucket = client.bucket('titanic_clement7991')
    blob = bucket.blob('cloud_train.csv')
    local_path=os.path.join('titanic/data','clean_train.csv')
    blob.download_to_filename(local_path)


    # create dataframe from cloud data
    config = Config()
    data = pd.read_csv(config.CLEAN_TRAIN_DATA) # from training set, create dataframe

    X = data.drop(columns=['Survived']) # create X
    y = data['Survived'] # create y

    # preprocess dataframe
    pipeline=pipeline_train()
    X_preproc_df = pipeline.fit_transform(X)

    print('Data preprocessed')

    return X_preproc_df, y

def preprocess_pred(X_pred=None):
    ''' returns preprocessed prediction dataframe'''

    pipeline=pipeline_pred()

    X_pred_preproc_df=pipeline.fit_transform(X_pred)
    # X_pred_preproc_df = pd.DataFrame(X_pred_preproc)

    return X_pred_preproc_df


### MODEL ###
def train_rfc():
    ''' returns a fitted Random Forest Classifier model '''
    X_preproc_df, y = preprocess_data()

    # initialize and fit model
    model=init_rfc()
    model.fit(X_preproc_df, y)

    # print('Model trained')

    return model

# Prediction based on test_data for now
def pred(X_pred : pd.DataFrame=None):
    ''' returns the recommendation based on the prediction made '''

    # Extract a row of the test set as X_pred in case of absence of prediction dataframe
    if X_pred is None:
        return 'Please provide information necessary to the prediction'

    # preprocess X_pred
    X_pred_preproc_df = preprocess_pred(X_pred.iloc[[0]])

    # instanciate model
    model = train_rfc()

    # predict
    y_pred = model.predict(X_pred_preproc_df)

    # translate into a recommendation the prediction
    if y_pred == 0:
        print('Whatever you do, do not board!')
    else:
        print('Welcome aboard!')
