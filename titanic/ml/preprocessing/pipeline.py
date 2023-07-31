### IMPORTS ###
from sklearn.pipeline import Pipeline
from titanic.ml.preprocessing.functions import Dropper, EncoderPred,Imputer,Encoder,Scaler

### PIPELINES ###

def pipeline_train():
    ''' Returns a preprocessing pipeline for the train set '''
    pipeline = Pipeline([
        ('Dropper', Dropper()),
        ('Imputer', Imputer()),
        ('Encoder', Encoder()),
        ('Scaler', Scaler())
    ])

    return pipeline


def pipeline_pred():
    ''' Returns a preprocessing pipeline for the data obtained from api'''
    pipeline_pred = Pipeline([
        ('Encoder', EncoderPred()),
        ('Scaler', Scaler())
    ])

    return pipeline_pred
