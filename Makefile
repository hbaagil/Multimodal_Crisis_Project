.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y crisis_helper || :
	@pip install -e .

preprocess_train_validate:
	python -c 'from crisis_helper.interface.main_local import preprocess_train_validate; preprocess_train_validate()'

#run_train:
#	python -c 'from crisis_helper.interface.main import train; train()'

run_pred_bin:
	python -c 'from crisis_helper.interface.main_local import pred_bin; pred_bin()'

run_pred_multi:
	python -c 'from crisis_helper.interface.main_local import pred_multiclass; pred_multiclass()'


#run_evaluate:
#	python -c 'from crisis_helper.interface.main import evaluate; evaluate()'

run_all: preprocess_train_validate run_pred_multi

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

#======================#
#         Docker       #
#======================#

# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(GCR_IMAGE):local .

docker_run_local:
	docker run \
    		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
        --env-file .env \
        $(GCR_IMAGE):local

docker_run_local_interactively:
	docker run -it \
        -e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
        --env-file .env \
        $(GCR_IMAGE):local \
        bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

docker_build:
	docker build \
        --platform linux/amd64 \
        -t $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod .

# Alternative if previous doesn´t work. Needs additional setup.
# Probably don´t need this. Used to build arm on linux amd64
docker_build_alternative:
	docker buildx build --load \
        --platform linux/amd64 \
        -t $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod .

docker_run:
	docker run \
        --platform linux/amd64 \
        -e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
        --env-file .env \
        $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod

docker_run_interactively:
	docker run -it \
        --platform linux/amd64 \
        -e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
        --env-file .env \
        $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod \
        bash

# Push and deploy to cloud

docker_push:
	docker push $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod

docker_deploy:
	gcloud run deploy \
        --project $(PROJECT_ID) \
        --image $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(DOCKER_IMAGE_NAME):prod \
        --platform managed \
        --region europe-west1 \
        --env-vars-file .env.yaml
