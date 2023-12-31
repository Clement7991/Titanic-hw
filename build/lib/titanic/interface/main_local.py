### IMPORTS ###
import pandas as pd
from config import Config
from titanic.ml.preprocessing.pipeline import pipeline_train, pipeline_pred
from titanic.ml.model.model_rfc import init_rfc
from sklearn.impute import SimpleImputer

###  ###
def preprocess_data():
    ''' returns preprocessed features and labels for training set'''

    config = Config()
    data = pd.read_csv(config.TRAIN_DATA) # from training set, create dataframe

    X = data.drop(columns=['Survived']) # create X
    y = data['Survived'] # create y

    # preprocess dataframe
    pipeline=pipeline_train()
    X_preproc = pipeline.fit_transform(X)
    X_preproc_df = pd.DataFrame(X_preproc)

    # print('Data preprocessed')

    return X_preproc_df, y

def preprocess_pred(X_pred=None):
    ''' returns preprocessed prediction dataframe'''

    pipeline=pipeline_pred()

    X_pred_preproc=pipeline.fit_transform(X_pred)
    X_pred_preproc_df = pd.DataFrame(X_pred_preproc)

    imputer=SimpleImputer(strategy='mean')
    X_pred_preproc_df_vf=pd.DataFrame(imputer.fit_transform(X_pred_preproc_df))

    return X_pred_preproc_df_vf


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
        config = Config()
        test_data = pd.read_csv(config.TEST_DATA)
        X_pred=test_data

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
