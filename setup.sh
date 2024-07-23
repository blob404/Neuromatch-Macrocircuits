#!/bin/bash

# Install micromamba (or use your preferred conda implementation)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Build a new virtual Python environment from the text file specification
# (Confirmed working on macOS Sonoma 14.4 using native osx-arm64 distributions)
micromamba create -n nma2024 -f ./deps -c conda-forge

# Install neuromatch distribution of tonic RL package
micromamba run -n nma2024 git clone https://github.com/neuromatch/tonic
mv ./tonic/data . # Data folder contains pre-trained notebook models; this folder structure should work for running the script
micromamba run -n nma2024 pip3 install -e ./tonic
