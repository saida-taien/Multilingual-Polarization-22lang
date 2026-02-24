# Multilingual Polarization Detection  
SemEval 2026 Task 9 – Subtask 1

This repository contains the test-phase code for our multilingual polarization detection system.  
The system was developed and evaluated across 22 languages using transformer-based models and an ensemble strategy.

------------------------------------------------------------

## Repository Structure

XLM-R/
    Language-specific scripts for XLM-R model
    (e.g., amh.py, ben.py, eng.py, etc.)

mDeBERTa/
    Language-specific scripts for mDeBERTa model
    (e.g., amh.py, ben.py, eng.py, etc.)

Ensemble/
    ensemble.py   → Generates final submission files
    utils.py      → Helper functions for ensemble

requirements.txt
    Required Python dependencies

------------------------------------------------------------

## How to Run

Step 1: Generate probability files for each language

Run XLM-R model:
    cd XLM-R
    python amh.py

Run mDeBERTa model:
    cd mDeBERTa
    python amh.py

(Replace "amh" with other language codes accordingly.)

This will generate:
    pred_LANG_xlm_probs.csv
    pred_LANG_mdeberta_probs.csv

Place all probability files inside the "probabilities/" folder.

------------------------------------------------------------

Step 2: Run Ensemble

    cd Ensemble
    python ensemble.py

This will:
    - Load probability files
    - Average predictions
    - Generate final submission files:
        pred_LANG.csv
    - Automatically download them (if running in Colab)

------------------------------------------------------------

## Requirements

Install dependencies using:

    pip install -r requirements.txt

------------------------------------------------------------

## Notes

- Only test-phase code is included in this repository.
- Dev/experimental scripts are not included.
- Ensure file paths match your local or Colab directory structure.
- All scripts assume binary classification (labels: 0, 1).

------------------------------------------------------------