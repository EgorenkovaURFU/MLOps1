#!/bin/bash

# Get Jenkins path
JENKINS_HOME = "$JENKINS_HOME"

# Catalog path
BUILD_DIR = "/build"

# Create catalog "build" in the root
mkdir -p "BUILD_DIR" || { echo "Error during create 'build' catalog"; exit 1;}

# Check MLOps dir 
if [ ! -d "$JENKINS_HOME/workspace/MLOps1"]; then
    echo "There is no MLOps catalog in Jenkins Home dir"
    exit 1
fi

# Copy the files and catalogs from $JENKINS_HOME/workspace/MLOps to 'build' catalog
cp -R "$JENKINS_HOME/workspace/MLOps1" "$BUILD_DIR" || { echo "Error during copy MLOps"; exit 1; }

# Create venv
python3 -m venv "$BUILD_DIR/venv" || { echo "Error during create venv"; exit 1; }

# venv activate
source "$BUILD_DIR/venv/bin/activate" || { echo "Error during venv activateion"; exit 1; }

# install requirements
pip install -r "$BUILD_DIR/mlops2/requirements.txt" || { echo "Error during install requirements.txt"; exit 1; }

echo "Run data_creation.py"
python data_creation.py || { echo "Error during run data_creation.py"; exit 1; }

echo "Run model_preprocessing.py"
python model_preprocessing.py || { echo "Error during run model_preprocessing.py"; exit 1; }

echo "Run model_preporarion.py"
python model_preporarion.py || { echo "Error during run model_preporation.py"; exit 1; }

echo "Run model_testing.py"
python model_testing.py || { echo "Error during run model_testing.py"; exit 1; }



