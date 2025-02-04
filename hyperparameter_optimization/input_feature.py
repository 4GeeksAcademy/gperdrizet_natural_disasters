'''Main function to run input feature optimization experiment.'''

##############################################################
# Environment set-up #########################################
##############################################################

# Standard library imports
import os
import multiprocessing as mp

# PyPI imports
import pandas as pd

# Internal imports
import functions.data_manipulation as data_funcs
import functions.training as training_funcs
import functions.analysis as analysis_funcs

# Set GPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Set multiprocessing start method to allow CUDA use in the workers
try:
    mp.set_start_method('spawn')
except RuntimeError:
    print('Multiprocessing spawn method already set.')

##############################################################
# Experiment set-up ##########################################
##############################################################

# Incident features to test
incident_features=['incidents', 'log_incidents', 'linear_log_incidents']

# Other features to include in each test run
other_features=['month_sin', 'month_cos']

# Training details
data_file='../data/resampled_disaster_data_all.parquet'
model_type='regression'
window=24
num_training_workers=5
test_train_split_fraction=0.1
training_epochs=1000
learning_rate=0.01
l1_weight=0.0001
l2_weight=0.001

##############################################################
# Optimization run ###########################################
##############################################################

if __name__ == '__main__':

	# Read the data
	raw_data_df=pd.read_parquet(data_file)

	# Loop on the feature types
	for incident_feature in incident_features:

		# Build datasets
		training_data, testing_data, target_column_index=data_funcs.prep_data(
			raw_data_df,
			incident_feature,
			other_features,
			window,
			test_train_split_fraction
		)

		# Build models
		model_builder=mp.Process(
			target=training_funcs.make_model_ensemble,
			args=(
				training_data['states'],
				training_data['training_features'],
				'incident_feature',
				learning_rate,
				l1_weight,
				l2_weight,
				model_type
			)
		)

		model_builder.start()
		model_builder.join()

		# Train the models
		result=training_funcs.run_training_workers(
			model_type,
			'incident_feature',
			num_training_workers,
			training_data,
			training_epochs
		)

		print(f'Training run finished: {result}')
            
		# Plot the training curves
		if model_type == 'regression':

			plot=analysis_funcs.plot_regression_training_curves(
				training_data['states'],
				'incident_feature',
				incident_feature,
				model_type
			)

			plot.savefig(
				f'./results/plots/training_curves/incident_feature/{incident_feature}_{model_type}_window_{window}.jpg',
				dpi=300
			)

			plot.close()