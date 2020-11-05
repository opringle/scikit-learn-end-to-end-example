import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def train():
    df = pd.read_csv('./data/train.csv')
    train_df, test_df = train_test_split(df, test_size = 0.2)
    # select columns to use in the model
    cont_cols = ['LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt',
                 'MasVnrArea', 'BsmtFinSF1']
    cat_cols = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
        'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
        'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    ]
    # define preprocessing steps for categorical and continuous features
    numeric_transformer = Pipeline(steps=[
        # impute NAs with the median value for that feature in the training set
        ('imputer', SimpleImputer(strategy='median')),
        # scale features to 0 mean and unit variance
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        # create a category called 'missing' for NA categorical feature values
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # one hot encode categorical features
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # apply each preprocessing pipeline to it's respective cols
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cont_cols),
            ('cat', categorical_transformer, cat_cols)])
    # create full pipeline including preprocessing and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', GradientBoostingRegressor(
        n_estimators=1000
    ))])
    X_train = train_df[cont_cols + cat_cols]
    X_test = test_df[cont_cols + cat_cols]
    Y_train = train_df['SalePrice'].values
    Y_test = test_df['SalePrice'].values
    pipeline.fit(X_train, Y_train)
    Y_hat_train = pipeline.predict(X_train)
    Y_hat_test = pipeline.predict(X_test)
    train_mae = mean_absolute_error(Y_train, Y_hat_train)
    test_mae = mean_absolute_error(Y_test, Y_hat_test)
    logging.info("MAE train = ${:.2f}\tMAE test = ${:.2f}".format(
        train_mae,
        test_mae
    ))
    train_log_rmse = mean_squared_error(np.log(Y_train), np.log(Y_hat_train), squared=False)
    test_log_rmse = mean_squared_error(np.log(Y_test), np.log(Y_hat_test), squared=False)
    logging.info("log RMSE train = {:.4f}\tlog RMSE test = {:.4f}".format(
        train_log_rmse,
        test_log_rmse
    ))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    train()

