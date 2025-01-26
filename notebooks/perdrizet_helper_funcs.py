# Standard library imports
import os
import logging
from typing import Tuple

# PyPI imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import random as tf_random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall

# Random
random_state=315

# Fix Tensorflow's global random seed
tf_random.set_seed(random_state)

# Suppress warning and info messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

############################################################################
### Helper function to format data for LSTM training #######################
############################################################################

def make_time_course(data_df: pd.DataFrame) -> Tuple[list, list, list, list]:
    '''Chunks data by state and extracts features and labels with
    a one month offset for labels. Uses first 70% of each state time 
    course for training data and the last 30% for validation. Returns
    training and validation features and labels as lists along 
    with the list of states.'''

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
        state_df=data_df.loc[:,(state),:].copy()

        if len(state_df) > 50:

            # Make sure the state chunk is sorted by month and year
            state_df.sort_index(level='month', inplace=True)
            state_df.sort_index(level='year', inplace=True)

            # Take the first 70% of the data for training
            training_state_df=state_df.iloc[:int((len(state_df) * 0.7))]

            # Collect features and labels with one month offset
            state_training_features=[]
            state_training_labels=[]

            for i in range(len(training_state_df) - 1):
                state_training_features.append([training_state_df.iloc[i]])
                state_training_labels.append([training_state_df.iloc[i + 1]['incidents_binary']])

            # Take the last 30% of the data for validation
            validation_state_df=state_df.iloc[int((len(state_df) * 0.7)):]

            # Collect features and labels with one month offset
            state_validation_features=[]
            state_validation_labels=[]

            for i in range(len(validation_state_df) - 1):
                state_validation_features.append([validation_state_df.iloc[i]])
                state_validation_labels.append([validation_state_df.iloc[i + 1]['incidents_binary']])

            # Collect the training and validation features and labels
            training_features.append(np.array(state_training_features))
            training_labels.append(np.array(state_training_labels))
            validation_features.append(np.array(state_validation_features))
            validation_labels.append(np.array(state_validation_labels))

    return training_features, training_labels, validation_features, validation_labels, states


############################################################################
### Helper function to build LSTM model ####################################
############################################################################

def build_lstm(model_order: int, num_features: int, learning_rate: float=0.0001) -> Sequential:
    '''Builds and compiles LSTM model'''

    # Define and adapt a normalization layer.
    # norm_layer=keras.layers.Normalization()
    # norm_layer.adapt(training_features)

    # Set-up the L1L2 for the dense layers
    # regularizer=L1L2(l1=0.001, l2=0.01) # Last best state: 0.001, 0.01

    # Define the model
    model=Sequential()
    model.add(Input((model_order,num_features), name='input'))
    # model.add(norm_layer)
    model.add(LSTM(1024, return_sequences=True, name='LSTM.1'))
    model.add(LSTM(512, return_sequences=True, name='LSTM.2'))
    model.add(LSTM(256, name='LSTM.3'))
    model.add(Dense(256, activation='relu', name='dense.1'))#, kernel_regularizer=regularizer))
    model.add(Dense(128, activation='relu', name='dense.2'))#, kernel_regularizer=regularizer))
    model.add(Dense(32, activation='relu', name='dense.3'))#, kernel_regularizer=regularizer))
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Define the optimizer
    optimizer=Adam(learning_rate=learning_rate)

    # Compile the model, specifying the type of loss to use during training and any extra metrics to evaluate
    model.compile(
        loss=BinaryCrossentropy(name='cross-entropy'),
        optimizer=optimizer,
        metrics=[
            Recall(name='recall'),
            Precision(name='precision')
        ]
    )

    return model


############################################################################
### Helper function to LSTM train model ####################################
############################################################################

def train_lstm(
        model: Sequential,
        training_features: np.array,
        training_labels: np.array,
        testing_features: np.array,
        testing_labels: np.array,
        class_weight: dict,
        epochs: int,
        batch_size: int
):
    
    '''Does one LSTM training run'''

    # Train the model
    result=model.fit(
        training_features,
        training_labels,
        validation_data=(testing_features, testing_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        class_weight=class_weight
    )

    return result, model


############################################################################
### Helper function to calculate class weighting ###########################
############################################################################

def get_class_weights(scaled_training_df: pd.DataFrame) -> dict:
    '''Takes training labels as dict. calculates class weights according to tensorflow
    method. Returns as dict.'''

    # Calculate class weighting
    # Class weighting scheme suggested in: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    flat_training_labels=scaled_training_df['incidents_binary'].values
    pos_examples=sum(flat_training_labels)
    neg_examples=len(flat_training_labels) - pos_examples

    neg_class_weight=(1 / neg_examples) * (len(flat_training_labels) / 2.0)
    pos_class_weight=(1 / pos_examples) * (len(flat_training_labels) / 2.0)
    class_weight={0: neg_class_weight, 1: pos_class_weight}

    return class_weight


############################################################################
### Helper function to plot LSTM training run ##############################
############################################################################


def plot_single_training_run(training_results: list, num_states: int=58, training_epochs: int=1) -> plt:
    '''Takes a training results dictionary, plots it.'''

    print(f'Have: {len(training_results)} training results')

    # Collect individual training run results
    training_cross_entropy=[]
    validation_cross_entropy=[]
    training_precision=[]
    validation_precision=[]
    training_recall=[]
    validation_recall=[]

    for result in training_results:

        training_cross_entropy.extend(result.history['loss'])
        validation_cross_entropy.extend(result.history['val_loss'])
        training_precision.extend(result.history['precision'])
        validation_precision.extend(result.history['val_precision'])
        training_recall.extend(result.history['recall'])
        validation_recall.extend(result.history['val_recall'])

    # Collect iteration mean training results
    i=0
    iteration_middle_epoch=num_states // 2
    iteration_middle_epochs=[iteration_middle_epoch]
    mean_training_cross_entropy=[]
    mean_validation_cross_entropy=[]
    mean_training_precision=[]
    mean_validation_precision=[]
    mean_training_recall=[]
    mean_validation_recall=[]

    epochs_per_iteration=num_states*training_epochs

    print(f'Collected: {len(training_cross_entropy)} training results')

    while i < len(training_cross_entropy):

        mean_training_cross_entropy.append(sum(training_cross_entropy[i:i + epochs_per_iteration])/len(training_cross_entropy[i:i + epochs_per_iteration]))
        mean_validation_cross_entropy.append(sum(validation_cross_entropy[i:i + epochs_per_iteration])/len(validation_cross_entropy[i:i + epochs_per_iteration]))
        mean_training_precision.append(sum(training_precision[i:i + epochs_per_iteration])/len(training_precision[i:i + epochs_per_iteration]))
        mean_validation_precision.append(sum(validation_precision[i:i + num_states])/len(validation_precision[i:i + epochs_per_iteration]))
        mean_training_recall.append(sum(training_recall[i:i + epochs_per_iteration])/len(training_recall[i:i + epochs_per_iteration]))
        mean_validation_recall.append(sum(validation_recall[i:i + epochs_per_iteration])/len(validation_recall[i:i + epochs_per_iteration]))

        i+=epochs_per_iteration

    # Set-up a 1x3 figure for metrics
    _, axs=plt.subplots(1,3, figsize=(12,4))
    axs=axs.flatten()

    epochs=list(range(len(training_cross_entropy)))
    iteration_middle_epochs=list(range((epochs_per_iteration//2), len(training_cross_entropy) + epochs_per_iteration, epochs_per_iteration))

    print(f'Iteration middle epochs: {len(iteration_middle_epochs)}')
    print(f'Training mean cross entropy: {len(mean_training_cross_entropy)}')

    # Plot Loss
    axs[0].set_title('Training loss: binary cross-entropy')
    axs[0].plot(epochs, training_cross_entropy, alpha=0.3, color='tab:blue',  label='Training batches')
    axs[0].plot(epochs, validation_cross_entropy, alpha=0.3, color='tab:orange', label='Validation batches')
    axs[0].plot(iteration_middle_epochs, mean_training_cross_entropy, color='tab:blue', label='Training iteration mean')
    axs[0].plot(iteration_middle_epochs, mean_validation_cross_entropy, color='tab:orange', label='Validation iteration mean')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Binary cross-entropy')
    axs[0].legend(loc='upper right')

    # Plot MSE
    axs[1].set_title('Precision')
    axs[1].plot(epochs, training_precision, alpha=0.3, color='tab:blue')
    axs[1].plot(epochs, validation_precision, alpha=0.3, color='tab:orange')
    axs[1].plot(iteration_middle_epochs, mean_training_precision, color='tab:blue', label='Training iteration mean')
    axs[1].plot(iteration_middle_epochs, mean_validation_precision, color='tab:orange', label='Validation iteration mean')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Precision')

    # Plot MAE
    axs[2].set_title('Recall')
    axs[2].plot(epochs, training_recall, alpha=0.3, color='tab:blue')
    axs[2].plot(epochs, validation_recall, alpha=0.3, color='tab:orange')
    axs[2].plot(iteration_middle_epochs, mean_training_recall, color='tab:blue', label='Training iteration mean')
    axs[2].plot(iteration_middle_epochs, mean_validation_recall, color='tab:orange', label='Validation iteration mean')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Recall')

    # Show the plot
    plt.tight_layout()

    return plt
