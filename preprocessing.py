import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from tabulate import tabulate


ddi_fp = "drugbank\drugbank.tab"

ddi = pd.read_csv(ddi_fp, sep='\t')

kaggle_fp = "SMILES-Kaggle\chembl_22_clean_1576904_sorted_std_final.smi"

smiles = pd.read_csv(kaggle_fp, sep='\t')

drug_names_fp = "drugs.txt"

drug_names = pd.read_csv(drug_names_fp, sep='\t')

# structure and metadata

# missing data 

# anomalies 

# bias 

# distributions 

# categorical data 

# ethical considerations 

# alignment with goals 

# scalability 

# transformations 

# data encoding 

# predictive power 

# target variable 

# validation strategy 

# data leakage

# interpretability 

# limitations 
