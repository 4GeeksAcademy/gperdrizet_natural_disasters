'''State LSTM ensemble training functions to be used in hyperparameter optimization experiments.'''

# Standard library imports
import os
import logging
import pickle
import multiprocessing as mp
from pathlib import Path

# PyPI imports
import numpy as np
from tensorflow import random as tf_random
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import Precision, Recall, RootMeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.config.experimental import list_physical_devices, set_memory_growth

# Random
random_state=315

# Fix Tensorflow's global random seed
tf_random.set_seed(random_state)

# Suppress warning and info messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

############################################################################
### Main function to train LSTM state ensemble #############################
############################################################################

def run_training_workers(
        model_type: str,
        independent_var: str,
        num_training_workers: int,
        training_data: dict,
        training_epochs: int
) -> None:

    '''Runs n training workers to train each state model.'''

    # Set-up multiprocessing queues for input to workers
    mp_manager=mp.Manager()
    input_queue=mp_manager.Queue(-1)

    # Set-up training worker processes
    training_workers=[]
    for _ in range(num_training_workers,):
        training_workers.append(
            mp.Process(
                target=state_training_run,
                args=(
                    input_queue,
                    training_epochs,
                    independent_var,
                    model_type
                )
            )
        )

    # Start the training workers
    for worker in training_workers:
        worker.start()

    # Put the work in the input queue
    for i in range(len(training_data['training_features'])):

        # Build the workunit
        workunit={
            'status': 'Work',
            'state': training_data['states'][i],
            'training_features': training_data['training_features'][i],
            'training_labels': training_data['training_labels'][i],
            'validation_features': training_data['validation_features'][i],
            'validation_labels': training_data['validation_labels'][i]
        }

        # Submit the workunit
        input_queue.put(workunit)

    # Send a stop signal for each worker
    for _ in range(num_training_workers,):
        input_queue.put({'status': 'Done'})

    # Join and then close each score training process
    for worker in training_workers:
        worker.join()
        worker.close()

    return True


############################################################################
### Multiprocessing worker function to train LSTM on state #################
############################################################################

def state_training_run(
    input_queue: mp.Queue,
    training_epochs: int,
    independent_var: str,
    model_type: str
) -> None:

    '''Does LSTM training run on one state. Meant to be called in a multiprocessing worker.
    Load pre-compiled model from disk and trains it with checkpoint callback. Saves
    final training results.'''

    # Set memory growth flag for each visible GPU
    gpus=list_physical_devices('GPU')
    for gpu in gpus:
        set_memory_growth(gpu, True)

    # Set-up output directories
    checkpoint_dir=f'./models/{independent_var}'
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    training_results_dir=f'./results/data/{independent_var}'
    Path(training_results_dir).mkdir(parents=True, exist_ok=True)

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
            model_path=f'{checkpoint_dir}/{model_type}_{workunit["state"]}_naive.keras'
            checkpoint_filepath=f'{checkpoint_dir}/{model_type}_{workunit["state"]}_trained.keras'
            results_filepath=f'{training_results_dir}/{model_type}_{workunit["state"]}.pkl'

            # Load the model
            model=load_model(model_path)

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
### Helper function to build and save LSTM for each state ##################
############################################################################

def make_model_ensemble(
    states: list,
    training_features: np.array,
    independent_var: str,
    learning_rate: float,
    l1_weight: float,
    l2_weight: float,
    model_type: str
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
    model_checkpoint_dir=f'./models/{independent_var}'
    Path(model_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for state in states:
        model.save(f'{model_checkpoint_dir}/{model_type}_{state}_naive.keras')


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
