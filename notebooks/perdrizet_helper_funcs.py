# Standard library imports
import os
import pickle
import logging
import multiprocessing as mp
from typing import Tuple

# PyPI imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import random as tf_random
from tensorflow.config.experimental import list_physical_devices
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import Precision, Recall, RootMeanSquaredError 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import list_physical_devices, set_memory_growth

# Random
random_state=315

# Fix Tensorflow's global random seed
tf_random.set_seed(random_state)

# Suppress warning and info messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# #############################################################################
# ### Helper function to format data for LSTM training on single time points ##
# #############################################################################

# def make_time_course(data_df: pd.DataFrame) -> Tuple[list, list, list, list]:
#     '''Chunks data by state and extracts features and labels with
#     a one month offset for labels. Uses first 70% of each state time 
#     course for training data and the last 30% for validation. Returns
#     training and validation features and labels as lists along 
#     with the list of states.'''

#     # Holders for results
#     training_features=[]
#     training_labels=[]
#     validation_features=[]
#     validation_labels=[]

#     # Get list of states
#     states=data_df.index.get_level_values('state').unique().tolist()

#     # Loop on states
#     for state in states:

#         # Extract the data for this state
#         state_df=data_df.loc[:,:,(state)].copy()

#         if len(state_df) > 50:

#             # Make sure the state chunk is sorted by month and year
#             state_df.sort_index(level='month', inplace=True)
#             state_df.sort_index(level='year', inplace=True)

#             # Take the first 70% of the data for training
#             training_state_df=state_df.iloc[:int((len(state_df) * 0.7))]

#             # Collect features and labels with one month offset
#             state_training_features=[]
#             state_training_labels=[]

#             for i in range(len(training_state_df) - 1):
#                 state_training_features.append([training_state_df.iloc[i]])
#                 state_training_labels.append([training_state_df.iloc[i + 1]['incidents_binary']])

#             # Take the last 30% of the data for validation
#             validation_state_df=state_df.iloc[int((len(state_df) * 0.7)):]

#             # Collect features and labels with one month offset
#             state_validation_features=[]
#             state_validation_labels=[]

#             for i in range(len(validation_state_df) - 1):
#                 state_validation_features.append([validation_state_df.iloc[i]])
#                 state_validation_labels.append(
#                     [validation_state_df.iloc[i + 1]['incidents_binary']]
#                 )

#             # Collect the training and validation features and labels
#             training_features.append(np.array(state_training_features))
#             training_labels.append(np.array(state_training_labels))
#             validation_features.append(np.array(state_validation_features))
#             validation_labels.append(np.array(state_validation_labels))

#     return training_features, training_labels, validation_features, validation_labels, states


#############################################################################
### Helper function to format data for LSTM training with time window #######
#############################################################################

def make_windowed_time_course(
        data_df: pd.DataFrame,
        target_column_index: int,
        window: int=5
) -> Tuple[list, list, list, list]:
    
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

    return training_features, training_labels, validation_features, validation_labels, states


############################################################################
### Helper function to build LSTM model ####################################
############################################################################

def build_lstm(
        model_order: int,
        num_features: int,
        learning_rate: float=0.0001,
        l1_weight: float=0.0001,
        l2_weight: float=0.001
) -> Sequential:
    '''Builds and compiles LSTM model'''

    # Set-up the L1L2 for the dense layers
    regularizer=L1L2(l1=l1_weight, l2=l2_weight) # Last best state: 0.001, 0.01

    # Define the model
    model=Sequential()
    model.add(Input((model_order,num_features), name='input'))
    model.add(LSTM(1024, return_sequences=True, name='LSTM.1'))
    model.add(LSTM(512, return_sequences=True, name='LSTM.2'))
    model.add(LSTM(256, name='LSTM.3'))
    model.add(Dense(256, activation='relu', name='dense.1', kernel_regularizer=regularizer))
    model.add(Dense(128, activation='relu', name='dense.2', kernel_regularizer=regularizer))
    model.add(Dense(32, activation='relu', name='dense.3', kernel_regularizer=regularizer))
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Define the optimizer
    optimizer=Adam(learning_rate=learning_rate)

    # Compile the model, specifying the type of loss to use during training 
    # and any extra metrics to evaluate
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
### Helper function to build and save state level LSTMs ####################
############################################################################

def make_state_models(
    states: list,
    training_features: np.array,
    learning_rate: float=0.001,
    l1_weight: float=0.0001,
    l2_weight: float=0.001,
    model_type: str='classification'
) -> None:

    '''Builds, compiles and saves naive LSTMs for each state in the
    state ensemble. Meant to be run in Multiprocessing worker process
    so that the CUDA context dies and rebases the GPU memory when done.'''

    # Set memory growth flag for each visible GPU
    gpus=list_physical_devices('GPU')
    for gpu in gpus:
        print(f'{gpu}: set_memory_growth==True')
        set_memory_growth(gpu, True)

    # Build and compile the model
    model=compile_state_lstm(
        training_features[0].shape[1],
        training_features[0].shape[2],
        learning_rate,
        l1_weight,
        l2_weight,
        model_type,
    )

    # Print out the model structure
    model.summary()

    # Save a copy of the untrained model for each state
    if model_type == 'classification':
        model_checkpoint_dir='../data/model_checkpoints/classification_state_LSTM_ensemble'

    elif model_type == 'regression':
        model_checkpoint_dir='../data/model_checkpoints/regression_state_LSTM_ensemble'

    else:
        print(f'Unrecognized model type: {model_type}')
        return None

    for state in states:
        model.save(f'{model_checkpoint_dir}/{state}_naive.keras')


############################################################################
### Helper function to compile LSTM for single state #######################
############################################################################

def compile_state_lstm(
        model_order: int,
        num_features: int,
        learning_rate: float,
        l1_weight: float,
        l2_weight: float,
        model_type: str
) -> Sequential:

    '''Builds and compiles LSTM classification or regression model for state ensemble.'''

    print(f'Compiling LSTM for state ensemble with L1={l1_weight}, L2={l2_weight}')

    # Set-up the L1L2 for the dense layers
    regularizer=L1L2(l1=l1_weight, l2=l2_weight)

    # Define the model
    model=Sequential()
    model.add(Input((model_order,num_features), name='input'))
    model.add(LSTM(512, name='LSTM'))
    model.add(Dense(256, activation='relu', name='dense.1', kernel_regularizer=regularizer))
    model.add(Dense(128, activation='relu', name='dense.2', kernel_regularizer=regularizer))
    model.add(Dense(32, activation='relu', name='dense.3', kernel_regularizer=regularizer))

    # Select the output, loss and metrics based on the type of model we are building
    if model_type == 'classification':
        model.add(Dense(1, activation='sigmoid', name='classification_output'))
        loss=BinaryCrossentropy(name='cross-entropy')
        metrics=[Recall(name='recall'), Precision(name='precision')]

    elif model_type == 'regression':
        model.add(Dense(1, activation='linear', name='regression_output'))
        loss=MeanSquaredError(name='MSE')
        metrics=[RootMeanSquaredError(name='RMSE')]

    else:
        print(f'Unrecognized model type: {model_type}')
        return None

    # Define the optimizer
    optimizer=Adam(learning_rate=learning_rate)

    # Compile the model, specifying the type of loss to use during training 
    # and any extra metrics to evaluate
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
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
        class_weight=class_weight,
    )

    return result, model


############################################################################
### Helper function to train LSTM model with model saving ##################
############################################################################

def train_lstm_with_checkpointing(
        model: Sequential,
        training_features: np.array,
        training_labels: np.array,
        testing_features: np.array,
        testing_labels: np.array,
        class_weight: dict,
        epochs: int,
        batch_size: int,
        checkpoint_filepath: str
):

    '''Does one LSTM training run, using a checkpoint callback to save the model
    during training.'''

    # Set up model saving checkpoint
    model_checkpoint_callback=ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Train the model

    if class_weight is None:
        result=model.fit(
            training_features,
            training_labels,
            validation_data=(testing_features, testing_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            class_weight=class_weight,
            callbacks=[model_checkpoint_callback],
            use_multiprocessing=False
        )
        
    else:
        result=model.fit(
            training_features,
            training_labels,
            validation_data=(testing_features, testing_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[model_checkpoint_callback],
            use_multiprocessing=False
        )


    return result


############################################################################
### Multiprocessing worker function to train LSTM on state #################
############################################################################

def state_training_run(
    input_queue: mp.Queue,
    training_epochs: int,
    model_type: str='classification'
) -> None:
    
    '''Does LSTM training run on one state. Meant to be called in a multiprocessing worker.
    Load pre-compiled model from disk and trains it with checkpoint callback. Saves
    final training results.'''

    # Set memory growth flag for each visible GPU
    gpus=list_physical_devices('GPU')
    for gpu in gpus:
        set_memory_growth(gpu, True)

    # Pick the output directories based on the type of model we are training
    if model_type == 'classification':
        checkpoint_dir='../data/model_checkpoints/classification_state_LSTM_ensemble'
        training_results_dir='../data/training_results/classification_state_LSTM_ensemble'

    elif model_type == 'regression':
        checkpoint_dir='../data/model_checkpoints/regression_state_LSTM_ensemble'
        training_results_dir='../data/training_results/regression_state_LSTM_ensemble'

    else:
        print(f'Unrecognized model type: {model_type}')
        return None

    # Loop forever
    while True:

        # Get a workunit from the queue
        workunit=input_queue.get()

        # Check for stop signal
        if workunit['status'] == 'Done':
            return

        # Do the work
        elif workunit['status'] == 'Work':

            # Set-up file paths for model checkpoints and training metrics
            checkpoint_filepath=f'{checkpoint_dir}/{workunit["state"]}_trained.keras'
            results_filepath=f'{training_results_dir}/{workunit["state"]}.pkl'

            # Get the class weights, if needed
            if model_type == 'classification':
                flat_training_labels=workunit['training_labels'].flatten()
                pos_examples=sum(flat_training_labels)
                neg_examples=len(flat_training_labels) - pos_examples

                neg_class_weight=(1 / neg_examples) * (len(flat_training_labels) / 2.0)
                pos_class_weight=(1 / pos_examples) * (len(flat_training_labels) / 2.0)
                class_weight={0: neg_class_weight, 1: pos_class_weight}

            elif model_type == 'regression':
                class_weight=None

            else:
                print(f'Unrecognized model type: {model_type}')
                return None

            # Load the model
            model_path=f'{checkpoint_dir}/{workunit["state"]}_naive.keras'
            model=load_model(model_path)

            # Train on this state
            results=train_lstm_with_checkpointing(
                model,
                workunit['training_features'],
                workunit['training_labels'],
                workunit['validation_features'],
                workunit['validation_labels'],
                class_weight,
                epochs=training_epochs,
                batch_size=workunit['training_features'].shape[0],
                checkpoint_filepath=checkpoint_filepath
            )

            with open(results_filepath, 'wb') as output_file:
                pickle.dump(results.history, output_file)


############################################################################
### Helper function to calculate class weighting ###########################
############################################################################

def get_class_weights(scaled_training_df: pd.DataFrame) -> dict:
    '''Takes training labels as dict. calculates class weights according to tensorflow
    method. Returns as dict.'''

    # Calculate class weighting
    # Class weighting scheme suggested in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    flat_training_labels=scaled_training_df['incidents_binary'].values
    pos_examples=sum(flat_training_labels)
    neg_examples=len(flat_training_labels) - pos_examples

    neg_class_weight=(1 / neg_examples) * (len(flat_training_labels) / 2.0)
    pos_class_weight=(1 / pos_examples) * (len(flat_training_labels) / 2.0)
    class_weight={0: neg_class_weight, 1: pos_class_weight}

    return class_weight


############################################################################
### Helper function to make state level predictions ########################
############################################################################

def make_state_predictions(
    state: str,
    state_features: np.array,
    results: list,
    model_type: str='classification'
) -> None:
    
    '''Helper function to make predictions for a specific state.
    Meant to be called in a multiprocessing worker process so that
    CUDA context does not survive. Returns predictions by adding them
    to a shared memory list as flat numpy array.'''

    predictions=[]

    # Set memory growth flag for each visible GPU
    gpus=list_physical_devices('GPU')
    for gpu in gpus:
        set_memory_growth(gpu, True)

    # Load the model
    if model_type == 'classification':
        model_path=f'../data/model_checkpoints/classification_state_LSTM_ensemble/{state}_trained.keras'

    elif model_type == 'regression':
        model_path=f'../data/model_checkpoints/regression_state_LSTM_ensemble/{state}_trained.keras'

    else:
        print(f'Unrecognized model type: {model_type}')
        return None

    model=load_model(model_path)

    for features in state_features:
        predictions.extend(model.predict(np.array([features]), verbose=0))

    results.append(np.array(predictions).flatten())


############################################################################
### Helper function to plot manually iterated LSTM training run ############
############################################################################

def plot_single_training_run(
        main_title: str,
        training_results: list,
        num_states: int=58,
        training_epochs: int=1
) -> plt:

    '''Takes a training results dictionary, plots it.'''

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
    mean_training_cross_entropy=[]
    mean_validation_cross_entropy=[]
    mean_training_precision=[]
    mean_validation_precision=[]
    mean_training_recall=[]
    mean_validation_recall=[]

    i=0
    epochs_per_iteration=num_states*training_epochs

    while i < len(training_cross_entropy):

        mean_training_cross_entropy.append(sum(training_cross_entropy[i:i + epochs_per_iteration])/len(training_cross_entropy[i:i + epochs_per_iteration]))
        mean_validation_cross_entropy.append(sum(validation_cross_entropy[i:i + epochs_per_iteration])/len(validation_cross_entropy[i:i + epochs_per_iteration]))
        mean_training_precision.append(sum(training_precision[i:i + epochs_per_iteration])/len(training_precision[i:i + epochs_per_iteration]))
        mean_validation_precision.append(sum(validation_precision[i:i + num_states])/len(validation_precision[i:i + epochs_per_iteration]))
        mean_training_recall.append(sum(training_recall[i:i + epochs_per_iteration])/len(training_recall[i:i + epochs_per_iteration]))
        mean_validation_recall.append(sum(validation_recall[i:i + epochs_per_iteration])/len(validation_recall[i:i + epochs_per_iteration]))

        i+=epochs_per_iteration

    # Set-up a 1x3 figure for metrics
    fig, axs=plt.subplots(1,3, figsize=(12,4))
    axs=axs.flatten()

    # Set up x vars for plotting
    epochs=list(range(len(training_cross_entropy)))
    iteration_middle_epochs=list(range((epochs_per_iteration//2), len(training_cross_entropy) + epochs_per_iteration, epochs_per_iteration))

    # Make sure the number of middle epochs and mean metrics match. Also, don't plot
    # The last one, always seems to droop on the graph
    iteration_middle_epochs=iteration_middle_epochs[:len(mean_training_cross_entropy) - 1]

    fig.suptitle(main_title)

    # Plot Loss
    axs[0].set_title('Training loss: binary cross-entropy')
    axs[0].plot(epochs, training_cross_entropy, alpha=0.3, color='tab:blue',  label='Training batches')
    axs[0].plot(epochs, validation_cross_entropy, alpha=0.3, color='tab:orange', label='Validation batches')
    axs[0].plot(iteration_middle_epochs, mean_training_cross_entropy[:-1], color='tab:blue', label='Training iteration mean')
    axs[0].plot(iteration_middle_epochs, mean_validation_cross_entropy[:-1], color='tab:orange', label='Validation iteration mean')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Binary cross-entropy')
    axs[0].legend(loc='upper right')

    # Plot MSE
    axs[1].set_title('Precision')
    axs[1].plot(epochs, training_precision, alpha=0.3, color='tab:blue')
    axs[1].plot(epochs, validation_precision, alpha=0.3, color='tab:orange')
    axs[1].plot(iteration_middle_epochs, mean_training_precision[:-1], color='tab:blue', label='Training iteration mean')
    axs[1].plot(iteration_middle_epochs, mean_validation_precision[:-1], color='tab:orange', label='Validation iteration mean')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Precision')

    # Plot MAE
    axs[2].set_title('Recall')
    axs[2].plot(epochs, training_recall, alpha=0.3, color='tab:blue')
    axs[2].plot(epochs, validation_recall, alpha=0.3, color='tab:orange')
    axs[2].plot(iteration_middle_epochs, mean_training_recall[:-1], color='tab:blue', label='Training iteration mean')
    axs[2].plot(iteration_middle_epochs, mean_validation_recall[:-1], color='tab:orange', label='Validation iteration mean')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Recall')

    plt.tight_layout()

    return plt


#############################################################################
### Helper function to plot per-state LSTM ensemble classification training #
#############################################################################

def plot_ensemble_training_run(states: list) -> plt:
    '''Loads state-level training histories from ensemble training run
    from disk and plots metrics as mean +/- standard deviation.'''

    # Collect the training history for each state
    state_dfs=[]

    for state in states:

        state_results_file=f'../data/training_results/classification_state_LSTM_ensemble/{state}.pkl'

        with open(state_results_file, 'rb') as input_file:
            state_data=pickle.load(input_file)

        state_data_df=pd.DataFrame.from_dict(state_data)
        state_data_df['epoch']=list(range(len(state_data_df)))

        state_dfs.append(state_data_df)

    results_df=pd.concat(state_dfs)

    # Get mean and standard deviation for each training metric across states at each epoch
    means=results_df.groupby(['epoch']).mean()
    standard_deviations=results_df.groupby(['epoch']).std()
    epochs=list(range(len(means)))

    # Set-up a 1x3 figure for metrics
    fig, axs=plt.subplots(1,3, figsize=(12,4))
    axs=axs.flatten()

    fig.suptitle('State LSTM ensemble training curves')

    # Plot Loss
    axs[0].set_title('Mean training loss: binary cross-entropy')
    axs[0].plot(epochs, means['loss'], color='tab:blue',  label='Training mean')
    axs[0].plot(epochs, means['val_loss'], color='tab:orange', label='Validation mean')
    axs[0].fill_between(epochs, means['loss'] + standard_deviations['loss'], means['loss'] - standard_deviations['loss'], color='tab:blue', alpha=0.3, label='Training standard deviation')
    axs[0].fill_between(epochs, means['val_loss'] + standard_deviations['val_loss'], means['val_loss'] - standard_deviations['val_loss'], alpha=0.3, color='tab:orange', label='Validation standard deviation')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Binary cross-entropy')
    axs[0].legend(loc='upper right')

    # Plot MSE
    axs[1].set_title('Precision')
    axs[1].plot(epochs, means['precision'], color='tab:blue',  label='Training mean')
    axs[1].plot(epochs, means['val_precision'], color='tab:orange', label='Validation mean')
    axs[1].fill_between(epochs, means['precision'] + standard_deviations['precision'], means['precision'] - standard_deviations['precision'], color='tab:blue', alpha=0.3, label='Training standard deviation')
    axs[1].fill_between(epochs, means['val_precision'] + standard_deviations['val_precision'], means['val_precision'] - standard_deviations['val_precision'], alpha=0.3, color='tab:orange', label='Validation standard deviation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Precision')

    # Plot MAE
    axs[2].set_title('Recall')
    axs[2].plot(epochs, means['recall'], color='tab:blue',  label='Training mean')
    axs[2].plot(epochs, means['val_recall'], color='tab:orange', label='Validation mean')
    axs[2].fill_between(epochs, means['recall'] + standard_deviations['recall'], means['recall'] - standard_deviations['recall'], color='tab:blue', alpha=0.3, label='Training standard deviation')
    axs[2].fill_between(epochs, means['val_recall'] + standard_deviations['val_recall'], means['val_recall'] - standard_deviations['val_recall'], alpha=0.3, color='tab:orange', label='Validation standard deviation')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Recall')

    plt.tight_layout()

    return plt


#############################################################################
### Helper function to plot per-state LSTM ensemble regression training run #
#############################################################################

def plot_regression_ensemble_training_run(states: list) -> plt:
    '''Loads state-level training histories from ensemble training run
    from disk and plots metrics as mean +/- standard deviation.'''

    # Collect the training history for each state
    state_dfs=[]

    for state in states:

        state_results_file=f'../data/training_results/regression_state_LSTM_ensemble/{state}.pkl'

        with open(state_results_file, 'rb') as input_file:
            state_data=pickle.load(input_file)

        state_data_df=pd.DataFrame.from_dict(state_data)
        state_data_df['epoch']=list(range(len(state_data_df)))

        state_dfs.append(state_data_df)

    results_df=pd.concat(state_dfs)

    # Get mean and standard deviation for each training metric across states at each epoch
    means=results_df.groupby(['epoch']).mean()
    standard_deviations=results_df.groupby(['epoch']).std()
    epochs=list(range(len(means)))

    # Set-up a 1x3 figure for metrics
    fig, axs=plt.subplots(1,2, figsize=(9,4))
    axs=axs.flatten()

    fig.suptitle('State LSTM regression ensemble training curves')

    # Plot Loss
    axs[0].set_title('Mean training loss: mean square error')
    axs[0].plot(epochs, means['loss'], color='tab:blue',  label='Training mean')
    axs[0].plot(epochs, means['val_loss'], color='tab:orange', label='Validation mean')
    axs[0].fill_between(epochs, means['loss'] + standard_deviations['loss'], means['loss'] - standard_deviations['loss'], color='tab:blue', alpha=0.3, label='Training standard deviation')
    axs[0].fill_between(epochs, means['val_loss'] + standard_deviations['val_loss'], means['val_loss'] - standard_deviations['val_loss'], alpha=0.3, color='tab:orange', label='Validation standard deviation')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('MSE')
    axs[0].legend(loc='upper right')

    # Plot MSE
    axs[1].set_title('Root mean square error')
    axs[1].plot(epochs, means['RMSE'], color='tab:blue',  label='Training mean')
    axs[1].plot(epochs, means['val_RMSE'], color='tab:orange', label='Validation mean')
    axs[1].fill_between(epochs, means['RMSE'] + standard_deviations['RMSE'], means['RMSE'] - standard_deviations['RMSE'], color='tab:blue', alpha=0.3, label='Training standard deviation')
    axs[1].fill_between(epochs, means['val_RMSE'] + standard_deviations['val_RMSE'], means['val_RMSE'] - standard_deviations['val_RMSE'], alpha=0.3, color='tab:orange', label='Validation standard deviation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('RMSE')

    plt.tight_layout()

    return plt