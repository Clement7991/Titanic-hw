### IMPORTS ###
import pandas as pd
from config import Config
from titanic.ml.preprocessing.pipeline import pipeline_init
from titanic.ml.model.model_rfc import init_rfc
from sklearn.impute import SimpleImputer

### DATA ###
def preprocess_data():
    config = Config()
    data = pd.read_csv(config.TRAIN_DATA)

    X = data.drop(columns=['Survived'])
    y = data['Survived']

    pipeline=pipeline_init()
    X_preproc = pipeline.fit_transform(X)
    X_preproc_df = pd.DataFrame(X_preproc)

    # print('Data preprocessed')

    return X_preproc_df, y

def preprocess_pred(X_pred=None):
    pipeline=pipeline_init()

    X_pred_preproc=pipeline.fit_transform(X_pred)
    X_pred_preproc_df = pd.DataFrame(X_pred_preproc)

    imputer=SimpleImputer(strategy='mean')
    X_pred_preproc_df_vf=pd.DataFrame(imputer.fit_transform(X_pred_preproc_df))

    return X_pred_preproc_df_vf


### MODEL ###
def train_rfc():
    X_preproc_df, y = preprocess_data()

    # initialize and fit model
    model=init_rfc()
    model.fit(X_preproc_df, y)

    # print('Model trained')

    return model

# Prediction based on test_data for now
def pred(X_pred : pd.DataFrame=None):
    if X_pred is None:
        config = Config()
        test_data = pd.read_csv(config.TEST_DATA)
        X_pred=test_data

    X_pred_preproc_df = preprocess_pred(X_pred)

    model = train_rfc()

    y_pred = model.predict(X_pred_preproc_df.iloc[[0]])


    if y_pred == 0:
        print('Whatever you do, do not board!')
    else:
        print('Welcome aboard!')
