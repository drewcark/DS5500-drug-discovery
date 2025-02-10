import csv
import math
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
import tensorflow as tf
import deepchem as dc

ddi_fp = "drugbank\drugbank.tab"
ddi = pd.read_csv(ddi_fp, sep='\t')

kaggle_fp = "SMILES-Kaggle\chembl_22_clean_1576904_sorted_std_final.smi"
smiles = pd.read_csv(kaggle_fp, sep='\t')

drug_names_fp = "drugs.txt"
drug_names = pd.read_csv(drug_names_fp, sep='\t')

ddi["Y"] = ddi["Y"].astype("category")
ddi["Map"] = ddi["Map"].astype("category")

interaction_counts = pd.DataFrame(ddi['Y'].value_counts().rename_axis('value').reset_index(name='count')).sort_values(by='count', ascending=False)
interaction_counts['row_num'] = interaction_counts.index + 1
interaction_counts['log_count'] = np.log(interaction_counts['count'])

interaction_types = ddi[['Y','Map']].drop_duplicates(subset=['Y'])

ddi = ddi.drop("Map",axis=1)

def delist(list_of_lists):
    list_of_strings = []
    for inner_list in list_of_lists:
        string = inner_list[0]
        list_of_strings.append(string)
    return list_of_strings

data = dc.data.NumpyDataset(X=ddi[['X1','X2']], y=ddi[['Y']])
df = data.to_dataframe()


featurizer = dc.feat.CircularFingerprint()
X_one = delist(df[["X1"]].values.tolist())
X_two = delist(df[["X2"]].values.tolist())
df["X1_feat"] = featurizer(X_one)
df["X2_feat"] = featurizer(X_two)