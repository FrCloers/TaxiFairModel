# imports
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None


    def splitting(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.15)

    def set_pipeline(self):
        '''returns a pipelined model'''
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
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipeline

    def run(self):
        '''returns a trained pipelined model'''
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(self.X_val)
        rmse = compute_rmse(y_pred, self.y_val)
        print(rmse)
        return rmse


if __name__ == "__main__":
    # store the data in a DataFrame
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    #instantiate Class
    trainer = Trainer(X, y)
    trainer.splitting()
    trainer.set_pipeline()
    # train
    trainer.run()
    # evaluate
    rmse = trainer.evaluate()
    print('rmse')
