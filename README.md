# GUIPilot

## Structure

This repository contains three components:
1. The **core** module (`/guipilot`).
3. The **datasets** module (`/dataset`), which records the dataset repositories.
2. The **experiments** module (`/experiments`), which supports the research questions 1-4 as presented in the paper.

The core GUIPilot module is organized as follows:

- `/agent`: Handles the action completion using a Vision-Language Model (VLM) agent
- `/matcher`: Pairs widgets across two different screens for comparison
- `/checker`: Detects bounding box, color, and text inconsistencies between widget pairs
- `/entities`: Defines Process, Screen, Widget, and Inconsistency entities used throughout the module
- `/models`: Contains OCR and widget detection models

## Setup
### Setup GUIPilot

Clone the repository and follow the steps below:

1. Create a conda environment.
    ```bash
    conda env create -f environment.yml
    conda activate guipilot
    ```

2. Install guipilot as a Python package.
    ```bash
    pip install .
    ```

### Setup Experiments

Each directory within `/experiments` includes a `README.md` file that provides detailed instructions on setting up the environment, preparing datasets, and running the experiment.