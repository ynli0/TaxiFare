# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
import pandas as pd

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()

        self.pipe.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        return compute_rmse(y_pred, y_test)



if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
