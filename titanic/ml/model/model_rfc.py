### IMPORTS ###
from sklearn.ensemble import RandomForestClassifier


### MODEL ###
def init_rfc():
    model = RandomForestClassifier(max_depth=None, min_samples_leaf=3, n_estimators=16)
    return model
