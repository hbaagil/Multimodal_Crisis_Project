.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y crisis_helper || :
	@pip install -e .

preprocess_train_validate:
	python -c 'from crisis_helper.interface.main_local import preprocess_train_validate; preprocess_train_validate()'

#run_train:
#	python -c 'from crisis_helper.interface.main import train; train()'

run_pred:
	python -c 'from crisis_helper.interface.main_local import pred; pred()'

#run_evaluate:
#	python -c 'from crisis_helper.interface.main import evaluate; evaluate()'

run_all: preprocess_train_validate run_pred

# run_workflow:
# 	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m crisis_helper.interface.workflow

run_api:
	@uvicorn crisis_helper.api.fast:app --reload --port 8008

##################### TESTS #####################


################### DATA SOURCES ACTIONS ################

ML_DIR=~/.crisis_helper/mlops/training_outputs

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir ~/.crisis_helper/mlops/training_outputs
	mkdir ~/.crisis_helper/mlops/training_outputs/metrics
	mkdir ~/.crisis_helper/mlops/training_outputs/models
	mkdir ~/.crisis_helper/mlops/training_outputs/models/vectorizer
	mkdir ~/.crisis_helper/mlops/training_outputs/models/model
	mkdir ~/.crisis_helper/mlops/training_outputs/params
