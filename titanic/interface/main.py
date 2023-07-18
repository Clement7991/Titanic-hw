### IMPORTS ###
import pandas as pd
from config import Config
from titanic.ml.preprocessing.pipeline import pipeline_init
from titanic.ml.model.model_rfc import init_rfc

### DATA ###
def preprocess_data():
    config = Config()
    data = pd.read_csv(config.TRAIN_DATA)

    X = data.drop(columns=['Survived'])
    y = data['Survived']

    pipeline=pipeline_init()
    X_preproc = pipeline.fit_transform(X)
    X_preproc_df = pd.DataFrame(X_preproc)

    print('Data preprocessed')

    return X_preproc_df, y

def preprocess_pred(X_pred):
    pipeline=pipeline_init()
    X_pred_preproc=pipeline.fit_transform(X_pred)
    return pd.DataFrame(X_pred_preproc)


### MODEL ###
def train_rfc():
    X_preproc_df, y = preprocess_data()

    # initialize and fit model
    model=init_rfc()
    model.fit(X_preproc_df, y)

    print('Model trained')
    return model

def predict(X_pred: pd.DataFrame=None):
    if X_pred is None:
        return "To predict, the model requires a DataFrame as input."

    model=train_rfc()
    X_pred_preproc_df = preprocess_pred(X_pred)

    y_pred = model.predict(X_pred_preproc_df)

    print('Prediction made')

    return y_pred

# if __name__ == '__main__':
#     preprocess_data()
#     preprocess_pred()
#     train_rfc()
#     predict()
