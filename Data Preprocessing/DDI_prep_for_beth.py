import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from tabulate import tabulate
import tensorflow as tf
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pyarrow.parquet as pq
import pyarrow as pa
import sqlalchemy
import pyodbc
warnings.filterwarnings("ignore", category=DeprecationWarning)

#quick function to turn a list of size 1 lists of strings into a list of strings, for later use
def delist(list_of_lists):
    list_of_strings = []
    for inner_list in list_of_lists:
        string = inner_list[0]
        list_of_strings.append(string)
    return list_of_strings

cf_featurizer = dc.feat.CircularFingerprint()

def circular_fingerprint(smiles):
    try:
        mol = cf_featurizer(smiles)
        return mol
    except Exception as e:
        print(f"Error fingerprinting {smiles}: {e}")
        return None  # Skipping invalid SMILES

#dimensionality reduction down to a certain number of features
#have X include cols to be able to rejoin with datasets if desired
def Dim_red(X, feature_num, batch_size, col_name):
	ipca = IncrementalPCA(n_components=feature_num)

	for i in range(0, X.shape[0], batch_size):
	    X_batch = X[i:i + batch_size]
	    ipca.partial_fit(X_batch)

	X_transformed = pd.DataFrame(ipca.transform(X))
	X_transformed['cols'] = X[col_name]

	#prove that X's shape has changed
	#print("Original shape:", X.shape)
	#print("Transformed shape:", X_transformed.shape)

	return(X_transformed)

#code from Beth Farr
def compute_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        # Computing Molecular Descriptors
        mol_wt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        return [mol_wt, logp, h_donors, h_acceptors, tpsa]

    except Exception as e:
        print(f"Error computing features for {smiles}: {e}")
        return None  # Skipping invalid SMILES

def ddi_featurize(df, X1_name, X2_name):
    df_cf = df
    df_cf['x1_cf'] = df[X1_name].apply(circular_fingerprint)
    df_cf['x2_cf'] = df[X2_name].apply(circular_fingerprint)
    df_cf['col'] = df_cf.index
    df_cf.dropna(subset=['x1_cf', 'x2_cf'], inplace=True)
    
    df_x1_cf = pd.DataFrame(delist(df_cf['x1_cf']))
    df_x1_cf.rename(columns=lambda x: "x1_cf_"+str(x+1), inplace=True)
    df_x1_cf['col'] = df_cf['col']
    
    df_x2_cf = pd.DataFrame(delist(df_cf['x2_cf']))
    df_x2_cf.rename(columns=lambda x: "x2_cf_"+str(x+1), inplace=True)
    df_x2_cf['col'] = df_cf['col']

    df_temp = df_cf.drop(['x1_cf', 'x2_cf'], axis=1)
    df_temp = df_temp.merge(df_x1_cf, on="col")
    df_temp = df_temp.merge(df_x2_cf, on="col")
    df_temp.dropna(inplace=True)

    df_temp['features_X1'] = df_temp['X1'].apply(compute_features)
    df_temp['features_X2'] = df_temp['X2'].apply(compute_features)
    df_temp = df_temp.dropna(subset=['features_X1', 'features_X2'])
    features_X1_df = pd.DataFrame(df_temp['features_X1'].tolist(), columns=['MolWt_X1', 'LogP_X1', 'NumHDonors_X1', 'NumHAcceptors_X1', 'TPSA_X1'])
    features_X1_df['col'] = df_temp['col']
    features_X2_df = pd.DataFrame(df_temp['features_X2'].tolist(), columns=['MolWt_X2', 'LogP_X2', 'NumHDonors_X2', 'NumHAcceptors_X2', 'TPSA_X2'])
    features_X2_df['col'] = df_temp['col']
    df_temp.drop(['features_X1', 'features_X2'], axis=1, inplace = True)
    
    df_final = pd.merge(df_temp, pd.merge(features_X1_df, features_X2_df,on='col'), on='col')
    df_final.drop(['col'], axis=1, inplace = True)

    return df_final

ddi_fp = "..\\Data Files\\drugbank\\drugbank.tab"
ddi = pd.read_csv(ddi_fp, sep='\t')

#kaggle_fp = "Data Files\\SMILES-Kaggle\\chembl_22_clean_1576904_sorted_std_final.smi"
#smiles = pd.read_csv(kaggle_fp, sep='\t')

ddi["Y"] = ddi["Y"].astype("category")
ddi["Map"] = ddi["Map"].astype("category")

#counting interaction types for potential later weighting
interaction_counts = pd.DataFrame(ddi['Y'].value_counts().rename_axis('y').reset_index(name='count')).sort_values(by='count', ascending=False)
interaction_counts['row_num'] = interaction_counts.index + 1
interaction_counts['log_count'] = np.log(interaction_counts['count'])

df_y_codes = pd.DataFrame(ddi['Y'].value_counts()).sort_values("count").reset_index()
df_y_codes = df_y_codes.head(20)
df_y_codes = df_y_codes.sort_values('Y').reset_index()
df_y_codes['int_reduced_code'] = df_y_codes.index
df_y_codes = df_y_codes.drop('count', axis=1)
df_y_codes['int_code'] = df_y_codes['Y']
df_y_codes.drop('Y', axis=1, inplace=True)


#listing longer explanations of interaction types for later use
interaction_types = ddi[['Y','Map']].drop_duplicates(subset=['Y'])
interaction_types.rename(columns={'Y': 'int_code', 'Map': 'int_desc'}, inplace=True)

# Database connection string (replace with your credentials)

engine = sqlalchemy.create_engine("mysql+pyodbc://admin:group7@DDI")

# Insert DataFrame into SQL table
#interaction_types.to_sql('interactions', engine, if_exists='replace', index=False)
#df_y_codes.to_sql('y_codes', engine, if_exists='replace', index=False)

print(interaction_types)
print(df_y_codes)