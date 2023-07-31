### IMPORTS ###
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from titanic.api.main import preprocess_pred, train_rfc

# instanciations
app = FastAPI() # instanciation of app
app.state.model = train_rfc() # to optimize app response

# Configuration of app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

### API RESPONSE ###

@app.get("/predict")
def predict(Pclass: int, Sex: str, Age:float, Sibs:int, Parch:int, Fare:float, Embarked:str):
    ''' Returns a recommendation on whether to board the Titanic or not. '''

    # constitution of prediction dataframe based on arguments passed
    X_pred=pd.DataFrame(dict(
        Pclass=[Pclass],
        Sex= [Sex],
        Age= [Age],
        SibSp=[Sibs],
        Parch=[Parch],
        Fare=[Fare],
        Embarked=[Embarked]
    ))

    # preprocessing of newly created prediction dataframe
    X_pred_preproc_df = preprocess_pred(X_pred)

    # instanciation of model
    model = app.state.model

    # prediction of survival
    y_pred = model.predict(X_pred_preproc_df)

    # translation of classification into recommendations
    if y_pred == 0:
        p = "Do not board whatever you do!"
    else:
        p = "Welcome aboard!"


    return p

# test: http://localhost:8000/predict?PassengerId=435&Pclass=473&Name=Clement%20Robin&Sex=male&Age=35&Sibs=0&Parch=0&Ticket=k25&Fare=12.5&Cabin=B42&Embarked=S


@app.get("/")
def root():
    ''' API test '''
    return {'message': 'Hello World'}
