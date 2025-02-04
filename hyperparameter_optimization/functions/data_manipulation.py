'''Collection of data manipulation functions for hyperparameter optimization experiments.'''

# Standard library imports
from typing import Tuple

# PyPI imports
import numpy as np
import pandas as pd


#############################################################################
### Helper function to format data for LSTM training with time window #######
#############################################################################

def prep_data(
        raw_data_df: pd.DataFrame,
        incident_feature: str,
        other_features: list,
        window: int,
        test_train_split_fraction: float
)-> Tuple[dict, dict, int]:
    
    '''Prepares data for LSTM training. Returns tuple containing a dictionary
    with testing and validation features and labels, and an int indicating the
    location of the target feature column.'''

    # Set-up dataframe with features we are interested in
    data_df, target_column_index=select_features(
        raw_data_df,
        incident_feature,
        other_features
    )

    # Train-test split the data
    training_df, testing_df=train_test_split(data_df, test_train_split_fraction)

    # Run training/validation splitting and formatting function on
    # the training data
    training_data=make_windowed_time_course(
        training_df,
        target_column_index,
        window
    )

    # Format the testing data
    testing_data=format_testing_data(
        testing_df,
        target_column_index,
        window
    )

    return training_data, testing_data, target_column_index


#############################################################################
### Helper function make dataframe for training #############################
#############################################################################

def select_features(
        raw_data_df: pd.DataFrame,
        incident_feature: str,
        other_features: list
) -> Tuple[pd.DataFrame, int]:
    
    '''Constructs dataframe with desired features, marks location of
    label. Returns tuple of dataframe and label column index.'''

    # Grab the features we are interested in and move to a new dataframe
    data_df=raw_data_df[[incident_feature] + other_features].copy()

    # Transfer the index
    data_df.set_index(raw_data_df.index)

    # Save the index of the target column
    target_column_index=data_df.columns.get_loc(incident_feature)

    return data_df, target_column_index


#############################################################################
### Helper function for sequential train-test splitting #####################
#############################################################################

def train_test_split(
        data_df: pd.DataFrame,
        test_train_split_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Splits data into sequential training and testing sets.
    Returns training and testing dataframes.'''

    # Get list of years
    years=data_df.index.get_level_values('year').unique().tolist()
    testing_years=int(len(years) * test_train_split_fraction)

    # Take last n years for testing data
    testing_df=data_df.loc[years[-testing_years:]]

    # Take the rest for training
    training_df=data_df.loc[years[:-testing_years]]

    return training_df, testing_df


#############################################################################
### Helper function to format training/validation data for LSTM #############
#############################################################################

def make_windowed_time_course(
        data_df: pd.DataFrame,
        target_column_index: int,
        window: int=5
) -> Tuple[list, list, list, list]:
    
    '''Chunks data by state and extracts features and labels with
    a one month offset for labels. Uses first 70% of each state time 
    course for training data and the last 30% for validation. Returns
    training and validation features and labels in a dictionary of lists 
    along with the list of states.'''

    # Holders for results
    training_features=[]
    training_labels=[]
    validation_features=[]
    validation_labels=[]

    # Get list of states
    states=data_df.index.get_level_values('state').unique().tolist()

    # Loop on states
    for state in states:

        # Extract the data for this state
        state_df=data_df.loc[:,:,(state)].copy()

        if len(state_df) > 50:

            # Make sure the state chunk is sorted by month and year
            state_df.sort_index(level='month', inplace=True)
            state_df.sort_index(level='year', inplace=True)

            # Take the first 70% of the data for training
            training_state_df=state_df.iloc[:int((len(state_df) * 0.7))]

            # Collect features and labels with one month offset
            state_training_features=[]
            state_training_labels=[]

            for i in range(len(training_state_df) - window - 1):
                state_training_features.append(training_state_df.iloc[i:i + window])
                state_training_labels.append([training_state_df.iloc[i + window + 1,target_column_index]])

            # Take the last 30% of the data for validation
            validation_state_df=state_df.iloc[int((len(state_df) * 0.7)):]

            # Collect features and labels with one month offset
            state_validation_features=[]
            state_validation_labels=[]

            for i in range(len(validation_state_df) - window - 1):
                state_validation_features.append(validation_state_df.iloc[i:i + window])
                state_validation_labels.append(
                    [validation_state_df.iloc[i + window + 1,target_column_index]]
                )

            # Collect the training and validation features and labels
            training_features.append(np.array(state_training_features))
            training_labels.append(np.array(state_training_labels))
            validation_features.append(np.array(state_validation_features))
            validation_labels.append(np.array(state_validation_labels))

        # Place results in a dictionary for easy handling later
        result={
            'training_features':training_features,
            'training_labels':training_labels,
            'validation_features':validation_features,
            'validation_labels':validation_labels,
            'states':states
        }

    return result

#############################################################################
### Helper function to format testing data for LSTM #########################
#############################################################################

def format_testing_data(
        testing_df: pd.DataFrame,
        target_column_index: int,
        window: int
) -> dict:
    
    '''Takes test set dataframe, target column index and window width, 
    returns dictionary containing formatted features and labels and
    list of states.'''

    testing_features=[]
    testing_labels=[]
    prediction_states=[]

    states=testing_df.index.get_level_values('state').unique().tolist()

    for state in states:

        state_df=testing_df.loc[:,:,(state)].copy()

        if len(state_df) > window + 1:

            state_testing_features=[]
            state_testing_labels=[]

            for i in range(len(state_df) - window -  1):

                state_testing_features.append(state_df.iloc[i:i + window])
                state_testing_labels.append(
                    [state_df.iloc[window + i + 1,target_column_index]]
                )

            testing_features.append(np.array(state_testing_features))
            testing_labels.append(np.array(state_testing_labels))
            prediction_states.append(state)

    result={
        'features':testing_features,
        'labels':testing_labels,
        'states':prediction_states
    }

    return result