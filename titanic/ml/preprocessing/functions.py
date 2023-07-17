### IMPORTS ###
import numpy as np
import pandas as pd


### FUNCTIONS ###
class Preprocessor:

    # add missing ages
    def add_missing_values(self, df):

        # add missing age values
        mean_age = round(df['Age'].mean())
        df['Age'].fillna(value=mean_age, inplace=True)

        # add missing embarked values
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'].fillna(value=mode_embarked, inplace=True)

        return df

    # drop cabin value
    def drop_columns(self, df):
        df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)
        return df
