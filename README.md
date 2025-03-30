# GUIPilot

[![arXiv](https://img.shields.io/badge/Paper-green)](http://linyun.info/publications/issta25.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://sites.google.com/view/guipilot/home)] [[Datasets](https://zenodo.org/records/15107436)] [[Models](https://huggingface.co/code-philia/GUIPilot)]

This is the official repository for the paper *"GUIPilot: A Consistency-based Mobile GUI Testing Approach for Detecting Application-specific Bugs"*, published at ISSTA 2025.

**GUIPilot** detects inconsistencies between mobile app designs and their implementations. It addresses two main types of inconsistencies: screen and process inconsistencies, using a combination of widget alignment and vision-language models. We’re continuously updating this repository. Stay tuned for more developments!

- Screen Inconsistency Detection:
    - Detects differences between the actual and expected UI appearance.
    - Converts the screen-matching problem into an optimizable widget alignment task.

- Process Inconsistency Detection:
    - Detects discrepancies between the actual and expected UI transitions after an action.
    - Translates natural language descriptions of transitions in mockups into stepwise actions (e.g., clicks, long-presses, text inputs).
    - Utilizes a vision-language model to infer actions on the real screen, ensuring that the expected transitions occur in the app.

## 📂 Structure

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

## ⚙️ Setup
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

## 📚 Citation
If you find our work useful, please consider citing our work.