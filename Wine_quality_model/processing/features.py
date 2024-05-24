from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1: int = 10):
        self.param1 = param1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Custom transformation logic
        return X
