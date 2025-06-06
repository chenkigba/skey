.DEFAULT_GOAL := usage
# Names of Docker image and container, change them to match your project
DOCKER_IMAGE := skey
DOCKER_CONTAINER := skey
CODE_DIRECTORY := skey
TEST_DIR := tests
GPUS=0

DOCKER_PARAMS= -it --rm --name=$(DOCKER_CONTAINER)
# Specify GPU device(s) to use. Comment out this line if you don't have GPUs available
DOCKER_PARAMS+= --gpus '"device=${GPUS}"'
# Mount music directories
DOCKER_PARAMS+=  -v /data/music:/data/music
DOCKER_PARAMS+=  -v /data/nfs/analysis/audio_database/beat_tracking:/data/nfs/analysis/audio_database/beat_tracking

# Run Docker container while mounting the local directory
DOCKER_RUN_MOUNT= docker run $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE)


DOCKER_IMAGE_NAME_FULL := $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

usage:
	@echo "Available commands:\n-----------"
	@echo "	build		Build the train Docker image"
	@echo "	run-bash	Run the train Docker image in a container after building it, and launches an interactive bash session in the container while mounting the current directory"
	@echo "	stop		Stop the train container if it is running"
	@echo "	logs		Display the logs of the train container"
	@echo "	exec		Launches a bash session in the train container (only if it is already running)"
	@echo "	poetry		Use poetry to modify 'pyproject.toml' and 'poetry.lock' files (e.g. 'make poetry add requests' to add the 'requests' package)"
	@echo "	qa			Check coding conventions using multiple tools"
	@echo "	qa-clean	Format your code using black and isort to fit coding conventions"


build:
	docker build -t $(DOCKER_IMAGE) -f Dockerfile.train .

run: build
	docker run $(DOCKER_PARAMS) $(DOCKER_IMAGE)

run-bash: build stop
	$(DOCKER_RUN_MOUNT) /bin/bash

stop:
	docker stop ${DOCKER_CONTAINER} || true && docker rm ${DOCKER_CONTAINER} || true

logs:
	docker logs -f $(DOCKER_CONTAINER)

exec:
	docker exec -it ${DOCKER_CONTAINER} /bin/bash

poetry:
	$(DOCKER_RUN_MOUNT) poetry $(filter-out $@,$(MAKECMDGOALS))
%:	# Avoid printing anything after executing the 'poetry' target
	@:

qa:
	poetry run pytest $(TEST_DIR)
# poetry run mypy $(CODE_DIRECTORY) $(TEST_DIR)
	poetry run ruff check --no-fix $(CODE_DIRECTORY) $(TEST_DIR)
	poetry run ruff format --check $(CODE_DIRECTORY) $(TEST_DIR)
	@echo "\nAll is good !\n"

qa-clean:
	poetry run ruff check --fix $(CODE_DIRECTORY) $(TEST_DIR)
	poetry run ruff format $(CODE_DIRECTORY) $(TEST_DIR)
