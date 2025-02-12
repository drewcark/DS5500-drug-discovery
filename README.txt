-------------------------------------------------------
AI in Drug Discovery: Predicting Drug-Drug Interactions
using SMILES codes for Safer Prescriptions

Beth Farr, Sreeja Vepa, Wren Warren

Northeastern DS5500 Spring 2025
-------------------------------------------------------

Project Description

This project aims to predict potential drug-drug interactions (DDIs) using Graph Neural networks (GNNs).
It uses Simplified Molecular Input Line Entry System (SMILES) codes, which captures chemical structure of a drug,
and drug interaction types to classify drug pairings into one of 86 different interaction types.
This predictive model has the potential to aid in patient safety, as users will be able to find the interaction type of any two drugs. 

Objectives and Goals 

- Predict drug-drug interactions from SMILES strings. 
- Explore the application of Graph Neural Networks (GNNs) for learning based on molecular structure. 
- Capture both topological and chemical properties of the drugs.
- Create an effective GUI for patient use. 
- Establish a SQL database to interact with the GUI. 

Setup and Instructions 

So far our code is not condensed in one file, as we have been working on different aspects individually,
but we plan to rectify that next. Relevant code can be found in DDI.ipynb, Data_Cleaning.ipynb, and preprocessing.py.

Data can be found in "drugbank.zip" (or "drugbank.csv.zip"),
"chembl_22_clean_1576904_sorted_std_final.csv.zip", and "drugs.txt"

RUNNING THIS CODE:
dependencies:
pip install -r requirements.txt
