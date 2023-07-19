import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from titanic.interface.main import pred, preprocess_pred, train_rfc


app = FastAPI()
app.state.model = train_rfc()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(PassengerId:int,Pclass: int,Name: str, Sex: str, Age:float, Sibs:int, Parch:int, Ticket:str, Fare:float, Cabin:str, Embarked:str):

    X_pred=pd.DataFrame(dict(
        PassengerId=[PassengerId],
        Pclass=[Pclass],
        Name=[Name],
        Sex= [Sex],
        Age= [Age],
        Sibs=[Sibs],
        Parch=[Parch],
        Ticket=[Ticket],
        Fare=[Fare],
        Cabin=[Cabin],
        Embarked=[Embarked]
    ))

    X_pred_preproc_df = preprocess_pred(X_pred)

    model = app.state.model

    y_pred = model.predict(X_pred_preproc_df)

    if y_pred == 0:
        p = "Do not board whatever you do!"
    else:
        p = "Welcome aboard!"


    return p

# test: http://localhost:8000/predict?PassengerId=892&Pclass=3&Name=Cl%C3%A9ment%20Robin&Sex=male&Age=26&SibSp=0&Parch=0&Ticket=330999&Fare=7.0000&Cabin=NaN&Embarked=S

@app.get("/")
def root():
    return {'message': 'Hello World'}
