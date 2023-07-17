### IMPORTS ###
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from titanic.ml.preprocessing.functions import Preprocessor


### PIPELINE ###

def pipeline_init():

    # Discriminating columns by dtype
    num_col = make_column_selector(dtype_include=['int64', 'float64'])
    obj_col = make_column_selector(dtype_include=['object'])

    # Pipeline creation
    num_transformer = Pipeline([('MinMax Scaler', MinMaxScaler())])
    obj_transformer = Pipeline([('Encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    preprocessor = Preprocessor()

    dropper = FunctionTransformer(preprocessor.drop_columns)
    imputer = FunctionTransformer(preprocessor.add_missing_values)

    basic_pipeline=ColumnTransformer(
        [('Imputer', imputer, ['Age', 'Embarked']),
        ('Dropper', dropper, ['Cabin', 'Ticket', 'Name']),
        ('MinMax Scaler', num_transformer, ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('Encoder', obj_transformer, ['Embarked', 'Sex'])]
    )

    return basic_pipeline
