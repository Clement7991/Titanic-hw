import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from titanic.interface.main import pred


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
    PassengerId : int,
    Pclass: int,
    Name: str,
    Sex: str,
    Age: int,
    SibSp: int,
    Parch: int,
    Ticket: str,
    Fare: float,
    Cabin: str,
    Embarked: str):

    x_pred=pd.DataFrame(dict(
        PassengerId=[PassengerId],
        Pclass=[Pclass],
        Name=[Name],
        Sex=[Sex],
        Age=[Age],
        Sibs=[SibSp],
        Parch=[Parch],
        Ticket=[Ticket],
        Fare=[Fare],
        Cabin=[Cabin],
        Embarked=[Embarked]
    ))

    return pred(x_pred)

# http://localhost:8000/predict?PassengerId=892&Pclass=3&Name=Cl%C3%A9ment%20Robin&Sex=male&Age=26&SibSp=0&Parch=0&Ticket=330999&Fare=7.0000&Cabin=NaN&Embarked=S
