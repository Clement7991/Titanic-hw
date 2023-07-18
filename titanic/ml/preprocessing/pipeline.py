### IMPORTS ###
from sklearn.pipeline import Pipeline
from titanic.ml.preprocessing.functions import Dropper,Imputer,Encoder,Scaler

### PIPELINE ###

def pipeline_init():
    # Create the pipeline
    pipeline = Pipeline([
        ('Dropper', Dropper()),
        ('Imputer', Imputer()),
        ('Encoder', Encoder()),
        ('Scaler', Scaler())
    ])

    return pipeline
