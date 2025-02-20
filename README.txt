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

cloning this repository:
  navigate to your preferrred location and open a terminal window, and enter:
    git clone https://github.com/drewcark/DS5500-drug-discovery.git

downloading and extracting data:
  data can be downloaded from https://tdcommons.ai/multi_pred_tasks/ddi and https://www.kaggle.com/datasets/art3mis/chembl22?resource=download,
  or data can be accessed from the Data Files section of this repository.

  processed datafiles (for "DDI_model_compare.ipynb" are in the main section: "DDI_eq_feat.parquet" and "DDI_red_feat.parquet"
    -these data files have been reduced down to ~10000 rows and ~178000 rows respectively

dependencies:
  pip install -r requirements.txt

to run the program:
  unzip the data files in Data Files
  open a terminal window in the main folder
  to observe preprocessing navigate to Data Preprocessing and run "python DDI_prep.py" to obtain data with molecular features and Morgan Fingerprints
	-(end product is "DDI_eq_feat.parquet" and "DDI_red_feat.parquet")
  navigate to Model Implementation, and:
    run "jupyter notebook feature_engineering.ipynb" for the feature engineering and simple GNN with individual features
    run "jupyter notebook DDI_model_compare.ipynb" for hyperparameter optimization of a deep feedforward neural network vs a graph neural network
  	-both of which use datasets with molecular features and Morgan Fingerprints
    run "jupyter notebook MolGraphGNN.ipynb" for Molecular Graphs and GNN
