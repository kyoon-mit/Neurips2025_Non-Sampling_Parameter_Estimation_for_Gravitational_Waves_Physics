#!/bin/bash

export CONDA_BASE=$(conda info --base)
export PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
