"""Helper module for EDA notebook to perform 
data cleaning and preprocessing"""

from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
from contextlib import contextmanager
from typing import Optional, Any
import sqlite3
from datetime import datetime

from scipy.stats import shapiro

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import textblob
from unidecode import unidecode
import xml.etree.ElementTree as ET

pd.plotting.register_matplotlib_converters()

def dummy_columns(df, feature_list):
    df_dummies = pd.get_dummies(df[feature_list]
                                #, drop_first=True
                                   )
    df = pd.concat([df, df_dummies], axis=1)
    return df


def populate_columns(xml_string):
    """Traverse the XML elements to populate column values"""
    root = ET.fromstring(xml_string)
    data = {}

    for element in root.iter():
        data[element.tag] = element.text
    return data



def df_xml(df: pd.DataFrame, feature: str) -> pd.DataFrame:

    def populate_columns_with_id(row):
        """Wrapper function to populate columns from XML with id"""
        xml_str = row[feature]  # Assuming the XML string is in the 'goal' column
        column_data = populate_columns(xml_str)
        return pd.Series(column_data)

    
    df = df[['id', feature]][~df[feature].isna()]
    df['match_id'] = df['id'].drop(columns='id')

    # Apply the populate_columns_with_id function to each row
    new_columns = df.apply(populate_columns_with_id, axis=1)

    # Combine the original DataFrame and the new columns
    result_df = pd.concat([df, new_columns], axis=1)

    # Drop the original 'goal' column if needed
    result_df = result_df.drop(columns=[feature]).dropna(axis=1, how='all')

    #Only 1 available value
    columns_to_drop = result_df.columns[result_df.nunique() == 1]
    result_df.drop(columns=columns_to_drop, inplace=True)
    result_df.drop(columns=['id', 'sortorder', 'value','event_incident_typefk', 'n', 'comment', 'goal_type'], errors='ignore', inplace=True)

    result_df.replace('', np.nan, inplace=True)
    result_df.replace('\n\t\t\t', np.nan, inplace=True)
    result_df.replace('\n\t\t\t\t', np.nan, inplace=True)
    result_df.dropna(axis=1, how='all', inplace=True)


    return result_df

def empty_rows_details(df):

    total = df.shape[0]
    for feature in df.columns:
        rows = df[feature].isna().sum()
        if (rows != 0): 
            print(f'{feature}: {total - rows} rows. Empty: {round(rows / total * 100, 1)}%')


def dtype_update(df: pd.DataFrame) -> pd.DataFrame:
    for feature in df.columns:
        if ('date' in feature.lower()) or ('birthday' in feature.lower()):
            try:
                # Convert to datetime and format as 'YYYY-MM-DD'
                df[feature] = pd.to_datetime(df[feature])
            except:
                pass

    
    return df

def remove_single_value_columns(df):
    # Get column names with only one unique value
    single_value_columns = df.columns[df.nunique() == 1]

    # Drop columns with only one unique value
    df = df.drop(columns=single_value_columns)
    df = df.dropna(axis=1, how='all')

    return df

def sql_download(cursor):
    '''Available Table names, features and observations'''
    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print('Available tables: \n')
    for table in tables:
        table_0 = table[0]
        table_name = cursor.execute(f"PRAGMA table_info({table_0})").fetchall()
        row_count = cursor.execute(
            f"SELECT COUNT(*) FROM {table_0}").fetchone()[0]

        print(f"Table '{table_0}'")
        print(f"Features: {[col[1] for col in table_name]}")
        print(f"Observations: {row_count} \n")

def csv_download(path: str) -> pd.DataFrame:
    """Download data, remove empty columns and capitalize the column names."""
    df = pd.read_csv(f"..\Capstone\Archive\{path}.csv", low_memory=False)

    df.replace('\r\n\t\t\t\t', np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df=dtype_update(df)

    return df


def first_look(df: pd.DataFrame) -> None:
    """Performs initial data set analysis."""

    print(f'Column data types:\n{df.dtypes}\n')
    print(f'Dataset has {df.shape[0]} observations and {df.shape[1]} features')
    print(f'Columns with NULL values: {df.columns[df.isna().any()].tolist()}')
    print(f'Dataset has {df.duplicated().sum()} duplicates')

def match_first_look(df):
    first_look(df)
    if 'type' in df.columns.to_list():
        print(f"Types available: {df['type'].unique()}")


def match_details(df):
    
    df.drop(columns=['id', 'n', 'sortorder', 'event_incident_typefk'], inplace=True)

    df = df[df['del']!=1].drop(columns=['del'])

    # Fiil in empty rows if match_id and player1 is knows
    if 'player1' in df.columns.to_list():
        condition = ~df['player1'].isna()
        df.loc[condition, 'team'] = df[condition].groupby(['match_id', 'player1'])['team'].fillna(method='bfill')
        df.loc[condition, 'team'] = df[condition].groupby(['match_id', 'player1'])['team'].fillna(method='ffill')

    df.dropna(axis=1, how='all', inplace=True)
    return df


def csv_download_match(path: str) -> pd.DataFrame:
    """Download data, remove empty columns and capitalize the column names."""
    if path == 'PositionReference':
        df = pd.read_csv(f"..\Capstone\Archive\{path}.csv").drop(columns =['role_x', 'role_y'])
    else:
        df = pd.read_csv(f"..\Capstone\Archive\{path}_detail.csv", low_memory=False)

        df.drop(columns=['id', 'n', 'sortorder', 'event_incident_typefk'], inplace=True) 
        #Drop spectators and venue with low column count
        if (df['type'].isin(['cross', 'corner'])).any():
            df.drop(columns=['spectators'], inplace=True)
        elif (df['type'] == 'foulcommit').any():
            df.drop(columns=['venue'], inplace=True) 

        df = df[df['del']!=1].drop(columns=['del']) 

        df.replace('\r\n\t\t\t\t', np.nan, inplace=True)

        if (df['type'] == 'goal').any():
            df.loc[df['comment'] != df['goal_type'], 'goal_type'] = df['comment']
            df.drop(columns=['comment'], inplace=True)
        elif (df['type'] == 'card').any():
            df.loc[df['comment'] != df['card_type'], 'card_type'] = df['comment']
            df.drop(columns=['comment'], inplace=True)
        

        # Fill in empty rows if match_id and player1 is knows
        if ('player1' in df.columns.to_list()) and df['team'].isna().any():
            condition = ~df['player1'].isna()
            df.loc[condition, 'team'] = df[condition].groupby(['match_id', 'player1'])['team'].fillna(method='bfill')
            df.loc[condition, 'team'] = df[condition].groupby(['match_id', 'player1'])['team'].fillna(method='ffill')
        


    df=dtype_update(df)
    df.dropna(axis=1, how='all', inplace=True)
    return df

def player_positions_extraction(df):
    result_df = pd.DataFrame()

    for prefix in ['home', 'away']:
        
        for i in range(11):
            subset_df = df[['id', 'league_id', 'win', 'season', 'stage', 'date', 'match_api_id', f'{prefix}_team_goal', f'{prefix}_team_api_id', f'{prefix}_player_X{i+1}', f'{prefix}_player_Y{i+1}', f'{prefix}_player_{i+1}']]
            subset_df.columns = ['id', 'league_id', 'win_team', 'season', 'stage', 'date', 'match_api_id', 'team_goal','team_api_id', 'X', 'Y', 'player']

            subset_df['type'] = prefix

            result_df = pd.concat([result_df, subset_df], axis=0, ignore_index=True)
    
    #result_df = result_df[~result_df['player'].isna()]

    result_df.loc[(result_df['win_team'] == result_df['type']), 'outcome'] = 'win'
    result_df.loc[(result_df['win_team'] == 'draw'), 'outcome'] = 'draw'   
    result_df.loc[(result_df['win_team'] != result_df['type']) & (result_df['win_team'] != 'draw'), 'outcome'] = 'loss'


    result_df['date'] = pd.to_datetime(result_df['date'])
    #result_df.drop(columns=['win_team'], inplace=True)
   
    return result_df


def remove_empty_dupes(df: pd.DataFrame, subset_cols: list) -> pd.DataFrame:
    # Find duplicated rows based on subset_cols
    duplicated_subset = df[df.duplicated(subset=subset_cols, keep=False)]

    # Filter rows where all non-subset columns are empty in duplicated subset
    non_subset_cols = df.columns.difference(subset_cols)
    filtered_rows = duplicated_subset.dropna(subset=non_subset_cols, how='all')

    # Get rows with a count of 1 for the combination of subset_cols
    count_one_combinations = df[~df.duplicated(subset=subset_cols, keep=False)]

    # Concatenate count one combinations with filtered rows
    joined_df = pd.concat([count_one_combinations, filtered_rows], ignore_index=True)
    
    return joined_df

def boxplot_per_feature(df: pd.DataFrame, feature_list: list, partition_feature: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    max_y_value = df[feature_list].max().max()+1

    for i, feature in enumerate(feature_list):
        sns.boxplot(data=df, y=feature, x=partition_feature, ax=axes[i])
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90) 
        #axes[i].set_ylim(bottom=-0.5, top=11)
        axes[i].set_ylim(bottom=-0.5, top=max_y_value) 

    fig.suptitle(f'Distribution against {partition_feature} ', y=1.02)
    plt.show()

def distribution_check(df: pd.DataFrame) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    sample_size = 1000

    for feature in df.columns:

        if df[feature].dtype == 'object': pass

        else:

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))  # Creating subplots

            print(f'{feature.capitalize()}')

            # Outlier check (Box plot)
            df.boxplot(column=feature, ax=axes[0])
            # sns.boxplot(data=df, y=feature, x='Quality', ax=axes[0])
            axes[0].set_title(
                f'{feature.capitalize()} ranges from {df[feature].min()} to {df[feature].max()}')
            
            sns.histplot(data=df, x=feature, kde=True, bins=20, ax=axes[1])
            axes[1].set_title(f'Distribution of {feature.capitalize()}')

            # Normality check (QQ plot).
            sm.qqplot(df[feature].dropna(), line='s', ax=axes[2])

            #sm.qqplot(np.random.choice(df[feature].dropna(), size=sample_size, replace=False), line='s', ax=axes[2])
            axes[2].set_title(f'Q-Q plot of {feature.capitalize()}')

            plt.tight_layout()
            plt.show()

def class_distribution(df: pd.DataFrame, feature_list: list) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))  # Creating subplots

    for i, feature in enumerate([item for item in feature_list if not item.endswith('Class')]):

        sns.histplot(data=df, x=feature, hue=feature+'Class', bins=40, ax=axes[i], palette='Set2')
        legend = axes[i].get_legend()
        legend.set_title('Class')


    plt.tight_layout()
    plt.show()


            
def distribution(df: pd.DataFrame, subset: list) -> None:
    for feature in df[subset]:
        # Perform Shapiro-Wilk test
        test_statistic, p_value = shapiro(df[feature])

        # Output results
        print(f"Test Statistic: {test_statistic}")
        print(f"P-value: {p_value}")

        # Check for normality at a 0.05 level
        alpha = 0.05
        if p_value < alpha:
            print("The data does not follow a normal distribution.")
        else:
            print("The data follows a normal distribution.")

def position_range_update(df: pd.DataFrame):
    x_range = [1, 9]
    y_range = [1, 11]

    def min_max_scaling(column, new_min, new_max):
        old_min = column.min()
        old_max = column.max()
        column_scaled = new_min + ((column - old_min) * (new_max - new_min)) / (old_max - old_min)
        return round(column_scaled, 0)

    columns_to_rescale = ['pos_x', 'pos_y']

    for column in columns_to_rescale:
        df[column] = min_max_scaling(df[column], x_range[0], x_range[1]) if column == 'pos_x' else min_max_scaling(df[column], y_range[0], y_range[1])


def heatmap(df: pd.DataFrame, name: str) -> None:
    # Plotting the heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True,
                cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title(f'Correlation {name.capitalize()} Attributes')

    plt.show()

def spearmans_corr(df: pd.DataFrame, features: list) -> None:

    for i_idx, i in enumerate(features):
        for j_idx, j in enumerate(features):
            if i_idx < j_idx:  # Avoid repeating pairs (i, j) and (j, i)
                correlation, p_value = stats.spearmanr(df[i].astype('category').cat.codes, df[j].astype('category').cat.codes)

                print(f"{i} + {j}. p-value = {p_value:.4f}: Spearman's correlation:  {correlation:.4f}")

                if abs(p_value) < alpha:
                    print("Reject null hypothesis")
                else:
                    print("Fail to reject null hypothesis")

def populate_columns(xml_string):
    """Traverse the XML elements to populate column values"""
    root = ET.fromstring(xml_string)
    data = {}

    for element in root.iter():
        data[element.tag] = element.text

    return data

def available_XML(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Get all available columns in the a column"""

    cureated_df = df[feature][~df[feature].isna()]

    # all_columns = set()
    # for xml_str in cureated_df:
    #     column_data = populate_columns(xml_str)
    #     all_columns.update(column_data.keys())

    # # Convert the set of columns to a list
    # all_columns_list = list(all_columns)
    # # Create a list of DataFrames and concatenate them

    dfs = []
    for xml_str in cureated_df:
        column_data = populate_columns(xml_str)
        dfs.append(pd.DataFrame([column_data]))

    # Create the final DataFrame by concatenating the list of DataFrames
    new_dataset = pd.concat(dfs, ignore_index=True)

    # Print the resulting DataFrame
    return new_dataset.dropna(axis=1, how='all')


def vif(df):
    """Calculating Variance Inflation Factor (VIF)."""
    df=df.select_dtypes(include=['int', 'float'])
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]

    return (vif)

def significance_t_test(df: pd.DataFrame, feature: str, change_feature: str, 
                        min_change_value: float, max_change_value: float) -> None:
    """Perform a t-test (sample size is small or when 
    the population standard deviation is unknown) and follows a normal distribution."""
    t_stat, p_value = stats.ttest_ind(df[df[change_feature] == min_change_value][feature],
                                      df[df[change_feature] == max_change_value][feature], equal_var=False)

    if p_value < alpha:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Reject null hypothesis')
    else:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Fail to reject null hypothesis')


def category_Mann_Whitney_U_test(df: pd.DataFrame, feature: str, change_feature: str, 
                                 min_change_value: float, max_change_value: float) -> None:
    """Category Mann-Whitney U test."""

    statistic, p_value = mannwhitneyu(df[df[change_feature] == min_change_value][feature],
                                      df[df[change_feature] == max_change_value][feature])

    if p_value < alpha:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Reject null hypothesis')
    else:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Fail to reject null hypothesis')


def confidence_intervals(data) -> None:
    """Calculate Confidence Intervals"""

    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

    critical_value = stats.norm.ppf((1 + confidence_level) / 2)

    standard_error = sample_std / np.sqrt(len(data))
    margin_of_error = critical_value * standard_error

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    print(f"Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")

def predictions(result, x_test) -> pd.Series:
    """Get predicted values for a linear model."""
    # Add a constant to the test features, same as per model creation
    x_test = sm.add_constant(x_test)

    # Make predictions on the test set
    y_pred = result.predict(x_test)
    y_pred_rounded = y_pred.round().astype(int)

    return y_pred, y_pred_rounded


def evaluation(y_pred_rounded, y_test) -> None:
    """Evaluating the model using metrics"""
    r_squared = r2_score(y_test, y_pred_rounded)
    mae = mean_absolute_error(y_test, y_pred_rounded)
    mse = mean_squared_error(y_test, y_pred_rounded)

    print(f"R-squared Validation: {r_squared:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

def feature_scaler(df, list):
    df['total'] = df[list].sum(axis=1) 

    # Scaling the 'total_attacking' column between 1 and 100
    scaler = MinMaxScaler(feature_range=(1, 100))
    df['total_scaled'] = MinMaxScaler(feature_range=(1, 100)).fit_transform(df[['total']]).astype(int)
    
    return df['total_scaled']

def visualization_fitted_model(df: pd.DataFrame, y_test, y_pred, feature) -> None:
    """Visualization for Linear fitted model, for Actual vs predicted values"""
    
    y_pred_rounded = y_pred.round().astype(int)
    y_test_rounded = y_test.round().astype(int)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_rounded, y=y_pred_rounded, alpha=0.1)

    # Adding the line f(x) = x
    mean = df[feature].mean().round()
    plt.axline((mean, mean), slope=1, color='red',
               linestyle='--', label='Diagonal')

    plt.xlabel(f'Actual {feature}')
    plt.ylabel(f'Predicted {feature}')
    plt.title(f'Actual vs. Predicted {feature}')
    plt.legend(loc='upper left')
    min = df[feature].min()
    max = df[feature].max()
    plt.xlim(min, max)
    plt.ylim(min, max)

    plt.show()


def confusion_matrix_visual(y_test, y_pred_rounded, new_labels: list) -> None:
    """Visualization for Confusion matrix on ordinal values"""
    conf_matrix = confusion_matrix(y_test, y_pred_rounded)

    new_labels = new_labels

    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=new_labels, yticklabels=new_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def residual_plot(y_test, y_pred) -> None:
    """Visualization for Residual Values"""
    # Align indices of y_test and y_pred before calculating residuals
    y_pred_rounded = y_pred.round().astype(int)

    y_test_aligned = y_test.reset_index(drop=True)
    y_pred_aligned = pd.Series(
        y_pred_rounded, index=y_test.index).reset_index(drop=True)

    residuals = y_test_aligned - y_pred_aligned

    # Create a DataFrame combining fitted values and residuals
    residual_df = pd.DataFrame(
        {'Fitted Values': y_pred_aligned, 'Residuals': residuals})

    # Residual plot
    sns.scatterplot(x='Fitted Values', y='Residuals', data=residual_df)
    plt.axhline(y=0, color='red', alpha=0.5, label='Residual Origin')
    plt.xlabel('Predicted values')
    plt.ylabel('Standartized Residuals')
    plt.title('Residuals')
    plt.legend(loc='upper right')

    plt.show()

def resampling(df, feature) -> pd.DataFrame:
    """Resampling data for Ordinal Feature, to match the maximum row count."""
    target_count = df[feature].value_counts().max()
    max_index = df[feature].value_counts().idxmax()
    resampled_wine = df[(df[feature] == max_index)]
    resampled = pd.DataFrame()

    for i in df[feature].unique():
        if i != max_index:
            resampled = resample(
                df[df[feature] == i], replace=True, n_samples=target_count, random_state=42)
            resampled_wine = pd.concat([resampled_wine, resampled], axis=0)
    return resampled_wine




"""Statistics"""
alpha = 0.05  # Significance level
confidence_level = 0.95
