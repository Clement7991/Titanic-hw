from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd

# Transformers

class Dropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=['Cabin', 'Ticket', 'Name'])
        return X

class Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        age_imputer = SimpleImputer(strategy='mean')
        X['Age'] = age_imputer.fit_transform(X[['Age']])

        embarked_mode = X['Embarked'].mode()
        X['Embarked'].fillna(value=embarked_mode[0], inplace=True)

        return X

class Encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        gen_dict = {'male': 0, 'female': 1}
        X['Sex'] = X['Sex'].map(gen_dict)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        tmp = np.array(X['Embarked']).reshape(-1, 1)
        encoded = encoder.fit_transform(tmp)
        encoded_columns = encoder.get_feature_names_out(['Embarked'])
        X_encoded = pd.DataFrame(encoded, columns=encoded_columns, index=X.index)
        X = pd.concat([X.drop(columns=['Embarked']), X_encoded], axis=1)

        return X

class EncoderPred(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        gen_dict = {'male': 0, 'female': 1}
        X['Sex'] = X['Sex'].map(gen_dict)

        if X['Embarked'].item() == 'S':
            X['Embarked_C'] = 0
            X['Embarked_Q'] = 0
            X['Embarked_S'] = 1
            X.drop(columns=['Embarked'], inplace=True)
        elif X['Embarked'].item() == 'C':
            X['Embarked_C'] = 1
            X['Embarked_Q'] = 0
            X['Embarked_S'] = 0
            X.drop(columns=['Embarked'], inplace=True)
        else:
            X['Embarked_C'] = 0
            X['Embarked_Q'] = 1
            X['Embarked_S'] = 0
            X.drop(columns=['Embarked'], inplace=True)

        return X

class Scaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
