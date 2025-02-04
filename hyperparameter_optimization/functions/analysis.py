'''Data analysis and plotting functions for hyperparameter optimization experiments.'''

# Standard library imports
import pickle

# PyPI imports
import pandas as pd
import matplotlib.pyplot as plt


#############################################################################
### Helper function to plot per-state LSTM ensemble classification training #
#############################################################################

def plot_classification_training_curves(states: list) -> plt:
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

def plot_regression_training_curves(
        states: list, 
        independent_var: str,
        independent_var_value: str,
        model_type: str
) -> plt:
    
    '''Loads state-level training histories from ensemble training run
    from disk and plots metrics as mean +/- standard deviation.'''

    # Collect the training history for each state
    state_dfs=[]

    for state in states:

        state_results_file=f'./results/data/{independent_var}/{model_type}_{state}.pkl'

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

    fig.suptitle(
        f'State LSTM regression ensemble training curves\n{independent_var}: {independent_var_value}')

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