# ------------------------------------------------------------------------------
# Data normalization
# ------------------------------------------------------------------------------
# Author:      Morteza Heidari
#
# Created:     11/20/2021
# ------------------------------------------------------------------------------
# different functions to normalize data, and balance imbalanced data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class DataNormalization():
    def __init__(self):
        pass

    def normalize_data(self, df, cols):
        """
        Normalize data
        """
        for col in cols:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df

    def normalize_data_by_group(self, df, cols, group_col):
        """
        Normalize data by group
        """
        for col in cols:
            df[col] = (df[col] - df[col].groupby(group_col).transform('min')) / (
                df[col].groupby(group_col).transform('max') - df[col].groupby(group_col).transform('min'))
        return df

    def normalize_data_by_group_with_mean_and_std(self, df, cols, group_col):
        """
        Normalize data by group with mean and std
        :param df: dataframe
        :param cols: columns to normalize
        :param group_col: group column
        :return: normalized dataframe
        """
        for col in cols:
            df[col] = (df[col] - df[col].groupby(group_col).transform('mean')) / (
                df[col].groupby(group_col).transform('std') + 1e-5)
        return df


class balancedata():
    def __init__(self):
        pass

    def balance_data(self, df, cols, group_col):
        """
        Balance data by group
        :param df: dataframe
        :param cols: columns to balance
        :param group_col: group column
        :return: balanced dataframe
        """
        for col in cols:
            df[col] = df[col].groupby(group_col).transform(lambda x: x.fillna(x.mean()))
        return df

    def balance_data_with_mean(self, df, cols, group_col):

        for col in cols:
            df[col] = df[col].groupby(group_col).transform(lambda x: x.fillna(x.mean()))
        return df

    def balance_data_with_mean_and_std(self, df, cols, group_col):

        for col in cols:
            df[col] = df[col].groupby(group_col).transform(lambda x: x.fillna(x.mean()))
            df[col] = (df[col] - df[col].groupby(group_col).transform('mean')) / (df[col].groupby(group_col).transform('std'))
        return df


def read_data(path):
    """
    Read data from csv file
    :param path: path to csv file
    :return: dataframe
    """
    df = pd.read_csv(path)
    return df


def save_data(df, path):
    """
    Save dataframe to csv file
    :param df: dataframe
    :param path: path to csv file
    """
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # read data
    df = read_data('./data/train.csv')
    # normalize data
    df = DataNormalization().normalize_data(df, ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'bp_hi', 'bp_lo', 'pregnant', 'insulin', 'bmi_cat', 'diabetes_cat', ])
    # balance data
    df = balancedata().balance_data(df, ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'bmi', 'bp_hi', 'bp_lo', 'pregnant', 'insulin', 'bmi_cat', 'diabetes_cat', ], 'diabetes_cat')
    # save data
    save_data(df, './data/train_balanced.csv')
