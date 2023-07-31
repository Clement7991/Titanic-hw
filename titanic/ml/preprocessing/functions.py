from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd

### TRANSFORMERS ###

class Dropper(BaseEstimator, TransformerMixin):
    ''' Transformer that returns a dataframe with unnecessary columns dropped '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'])

        return X

class Imputer(BaseEstimator, TransformerMixin):
    ''' Transformer to add missing values to the dataframe '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # For all ages missing, let's add the average age of passengers aboard
        age_imputer = SimpleImputer(strategy='mean')
        X['Age'] = age_imputer.fit_transform(X[['Age']])

        # For all port of embarcation missing, let's add the most frequent one
        embarked_mode = X['Embarked'].mode()
        X['Embarked'].fillna(value=embarked_mode[0], inplace=True)

        return X


class Encoder(BaseEstimator, TransformerMixin):
    ''' Transformer that returns a dataframe for model training with object features
    encoded. '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Let's replace male and female values by 0 and 1 respectively
        gen_dict = {'male': 0, 'female': 1}
        X['Sex'] = X['Sex'].map(gen_dict)

        # Let's One hot encode the port of embarcation information and then drop
        # the initial column
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        tmp = np.array(X['Embarked']).reshape(-1, 1)
        encoded = encoder.fit_transform(tmp)
        encoded_columns = encoder.get_feature_names_out()
        X_encoded = pd.DataFrame(encoded, columns=encoded_columns, index=X.index)
        X = pd.concat([X.drop(columns=['Embarked']), X_encoded], axis=1)

        return X

class EncoderPred(BaseEstimator, TransformerMixin):
    ''' Transformer that returns a prediction dataframe with object features encoded '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Let's replace male and female values by 0 and 1 respectively
        gen_dict = {'male': 0, 'female': 1}
        X['Sex'] = X['Sex'].map(gen_dict)

        # For a one row prediction dataframe to fit the pipeline, let's encode manually.
        if X['Embarked'].item() == 'S':
            X['Embarked_C'] = 0
            X['Embarked_Q'] = 0
            X['Embarked_S'] = 1

        elif X['Embarked'].item() == 'C':
            X['Embarked_C'] = 1
            X['Embarked_Q'] = 0
            X['Embarked_S'] = 0

        else:
            X['Embarked_C'] = 0
            X['Embarked_Q'] = 1
            X['Embarked_S'] = 0

        X.drop(columns=['Embarked'], inplace=True)

        return X

class Scaler(BaseEstimator, TransformerMixin):
    ''' Transformer that returns a dataframe whith values scaled between 0 and 1 '''
    def fit(self, X, y=None):
        return self

    # MinMaxScaler fit and transformation
    def transform(self, X):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X_scaled_df
