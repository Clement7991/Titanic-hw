### IMPORTS ###
from sklearn.pipeline import Pipeline
from titanic.ml.preprocessing.functions import Dropper, EncoderPred,Imputer,Encoder,Scaler

### PIPELINE ###

def pipeline_train():
    # Create the pipeline
    pipeline = Pipeline([
        ('Dropper', Dropper()),
        ('Imputer', Imputer()),
        ('Encoder', Encoder()),
        ('Scaler', Scaler())
    ])

    return pipeline

def pipeline_pred():
    pipeline = Pipeline([
        ('Dropper', Dropper()),
        ('Imputer', Imputer()),
        ('Encoder', EncoderPred()),
        ('Scaler', Scaler())
    ])

    return pipeline
