import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from typing import Tuple

# List for capturing MAE scores
scores = []


def read_data(path: str) -> pd.DataFrame:
    """Reads data from csv and returns a DataFrame"""
    return pd.read_csv(path, index_col='Id')


def get_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Remove rows with missing target, separates target from predictors, only uses numerical predictors, breaks off
        validation set from training set and returns, test, training and validations datasets"""
    X_full = read_data('./inputs/train.csv')
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    print(type(y))
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    X = X_full.select_dtypes(exclude=['object'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    return X_train, X_valid, y_train, y_valid


def get_test_data() -> pd.DataFrame:
    """Remove rows with missing target, separates target from predictors, only uses numerical predictors, breaks off
        validation set from training set and returns, test, training and validations datasets"""
    X_test_full = read_data('./inputs/test.csv')
    X_test = X_test_full.select_dtypes(exclude=['object'])

    return X_test


def plot_correlations(X: pd.DataFrame, y: pd.Series) -> None:
    """Plots the correlation of features with target"""
    for column in X.columns:
        plt.figure()
        plt.scatter(X[column], y)
        plt.title(f"{column} v Sales Price", fontsize=18)
        plt.xlabel(column, fontsize=14)
        plt.ylabel("Sales Price", fontsize=14)
        plt.subplots_adjust(left=0.15)
        plt.show()


def drop_incomplete_columns(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drops columns with missing values"""
    columns_with_missing_values = X_train.columns[X_train.isnull().any()].tolist()

    # Drop columns in training and validation data
    reduced_X_train = X_train.drop(columns_with_missing_values, axis=1)
    reduced_X_valid = X_valid.drop(columns_with_missing_values, axis=1)

    return reduced_X_train, reduced_X_valid


def imputation(X_train: pd.DataFrame, X_valid: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assigns values to missing data based on simple averaging strategies"""
    imputer = SimpleImputer(strategy=strategy)
    imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    return imputed_X_train, imputed_X_valid


def evaluate_dataset(X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series) \
                     -> Tuple[np.ndarray, float]:
    """Returns the model predictions and determines the mean absolute error of the model predictions against
       the actual observations"""
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)

    return preds, mae


def get_mae(strat_X_train: pd.DataFrame, strat_X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series,
            strategy: str) -> str:
    """Calculates MAE and adds it to list of MAE scores"""
    global scores
    strat_preds, strat_score = evaluate_dataset(strat_X_train, strat_X_valid, y_train, y_valid)
    scores.append(strat_score)
    message = f"\nMAE ({strategy}): {strat_score}"
    return message


# ----- GET DATA AND CHECK IT OUT -----
X_train, X_valid, y_train, y_valid = get_training_data()
X_test = get_test_data()

print(X_train.head())
print(X_train.shape)
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
plot_correlations(X_train, y_train)
# -------------------------------------

# ----- STRATEGY 1: Drop columns with missing data -----
strat1_X_train, strat1_X_valid = drop_incomplete_columns(X_train, X_valid)
print(get_mae(strat1_X_train, strat1_X_valid, y_train, y_valid, "Drop columns with missing data"))
# ----------------------------------------------------

# ----- STRATEGY 2: Imputation 'mean' strategy -----
strat2_X_train, strat2_X_valid = imputation(X_train, X_valid, 'mean')
print(get_mae(strat2_X_train, strat2_X_valid, y_train, y_valid, "Imputation 'mean' strategy"))
# ----------------------------------------------------

# ----- STRATEGY 3: Imputation 'median' strategy -----
strat3_X_train, strat3_X_valid = imputation(X_train, X_valid, 'median')
print(get_mae(strat3_X_train, strat3_X_valid, y_train, y_valid, "Imputation 'median' strategy"))
# ----------------------------------------------------

print(f"\nStrategy {scores.index(min(scores)) + 1} produces the lowest MAE")


# Preprocess test data and get test predictions
final_X_train, final_X_test = imputation(X_train, X_test, 'median')
chosen_model = RandomForestRegressor(n_estimators=100, random_state=0)
chosen_model.fit(final_X_train, y_train)
test_preds = chosen_model.predict(final_X_test)

