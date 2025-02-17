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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
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
import warnings
warnings.filterwarnings('ignore')

cf_featurizer = dc.feat.CircularFingerprint()

#quick function to turn a list of size 1 lists of strings into a list of strings, for later use
def delist(list_of_lists):
    list_of_strings = []
    for inner_list in list_of_lists:
        string = inner_list[0]
        list_of_strings.append(string)
    return list_of_strings

def circular_fingerprint(smiles):
    try:
        mol = cf_featurizer(smiles)
        return mol
    except Exception as e:
        print(f"Error fingerprinting {smiles}: {e}")
        return None  # Skipping invalid SMILES

#basic/rough neural network implementation

def calc_layers(X_size, Y_size):
    layers = [X_size+1]
    layer = 2
    while layer <= X_size:
        layer = int(layer * 2)
    layers.append(layer)
    if X_size > Y_size:
        while layer / 2 > Y_size and layer > 2:
            layer = layer / 2
            layers.append(int(layer))
    layers.append(Y_size)
    return layers
  
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

def DNN(X, Y, Epochs, batchsize, layernum):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=27)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #define layers and nodes in each layer
    input = len(X.columns)
    output = Y.shape[1]
    if layernum=='many':
        layers = calc_layers(input,output)
    elif type(layernum)==int:
        layers = [input]
        for i in range(layernum, 1, -1):
            layer = int(round((i * (input + output) / (layernum+1)), 0))
            if layer > output:
                layers.append(layer)
        layers.append(output)
    else:
        print(f"incorrect layernum {layernum}")
        return None

    print(f"Layers defined: {layers}")

    model = keras.models.Sequential()

    model.add(Dense(layers[0], activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    for layer_size in layers[1:-1]:
        model.add(Dense(layer_size, activation='relu'))
        
    model.add(Dense(layers[-1], activation='softmax'))
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy','AUC','precision','recall'])
    
    model.fit(X_train, y_train, epochs=Epochs, batch_size=batchsize, validation_split=0.1)
    
    full_loss, full_accuracy, full_AUC, full_precision, full_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {full_accuracy}")
    print(f"Test Loss: {full_loss}")
    return model


ddi_fp = "Data Files\\drugbank\\drugbank.tab"
ddi = pd.read_csv(ddi_fp, sep='\t')

kaggle_fp = "Data Files\\SMILES-Kaggle\\chembl_22_clean_1576904_sorted_std_final.smi"
smiles = pd.read_csv(kaggle_fp, sep='\t')

ddi["Y"] = ddi["Y"].astype("category")
ddi["Map"] = ddi["Map"].astype("category")

#counting interaction types for potential later weighting
interaction_counts = pd.DataFrame(ddi['Y'].value_counts().rename_axis('value').reset_index(name='count')).sort_values(by='count', ascending=False)
interaction_counts['row_num'] = interaction_counts.index + 1
interaction_counts['log_count'] = np.log(interaction_counts['count'])

#listing longer explanations of interaction types for later use
interaction_types = ddi[['Y','Map']].drop_duplicates(subset=['Y'])

#remove longer name of interaction type from main dataset
ddi = ddi.drop("Map",axis=1)

# counting drugs by number of mentions in database
old = pd.DataFrame()
old["total"] = ddi['ID1'].value_counts()
old = old.reset_index()
old.columns = ['ID', 'count'] 
new = pd.DataFrame()
new["total"] = ddi['ID2'].value_counts()
new = new.reset_index()
new.columns = ['ID', 'count'] 
drug_counts = pd.merge(old,new,how='outer',on='ID').fillna(0)
drug_counts['total'] = drug_counts['count_x'] + drug_counts['count_y']

drug_counts = drug_counts.sort_values(by='total')
drug_counts_one = pd.DataFrame(drug_counts[drug_counts['total']==1]['ID'])

#removing drugs only in database once
ddi_proc = ddi[ ~ddi['ID1'].isin(drug_counts_one['ID'])]
ddi_proc = ddi_proc[ ~ddi_proc['ID2'].isin(drug_counts_one['ID'])]

#removing one particular drug with a problematic SMILES code
ddi_proc = ddi_proc[ddi_proc['X1']!="OC1=CC=CC(=C1)C-1=C2\CCC(=N2)\C(=C2/N\C(\C=C2)=C(/C2=N/C(/C=C2)=C(\C2=CC=C\-1N2)C1=CC(O)=CC=C1)C1=CC(O)=CC=C1)\C1=CC(O)=CC=C1"]


#create main datasets

data = dc.data.NumpyDataset(X=ddi_proc[['X1','X2']], y=ddi[['Y']])
df = data.to_dataframe()
df = df.sample(frac=1).reset_index(drop=True)

X_one = delist(df[["X1"]].values.tolist())
X_two = delist(df[["X2"]].values.tolist())

#featurization

#other way to featurize a molecule
#cm_featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)

#whole dataset, run in discovery
df.rename(columns={'ids':'col'},  inplace=True)

df_cf = pd.DataFrame()
df_cf['x1_cf'] = df['X1'].apply(circular_fingerprint)
df_cf['x2_cf'] = df['X2'].apply(circular_fingerprint)
df_cf['col'] = df_cf.index
df_cf.dropna(subset=['x1_cf', 'x2_cf'], inplace=True)


df_x1_cf = pd.DataFrame(delist(df_cf['x1_cf']))
df_x1_cf.rename(columns=lambda x: "x1_cf_"+str(int(x)+1), inplace=True)
df_x1_cf['col'] = df_cf['col']


df_x2_cf = pd.DataFrame(delist(df_cf['x2_cf']))
df_x2_cf.rename(columns=lambda x: "x2_cf_"+str(int(x)+1), inplace=True)
df_x2_cf['col'] = df_cf['col']


df = df.merge(df_x1_cf, on="col")
df = df.merge(df_x2_cf, on="col")
df.dropna(inplace=True)


df['features_X1'] = df['X1'].apply(compute_features)
df['features_X2'] = df['X2'].apply(compute_features)
df = df.dropna(subset=['features_X1', 'features_X2'])


feat_X1_df = pd.DataFrame(df['features_X1'].tolist(), columns=['MolWt_X1', 'LogP_X1', 'NumHDonors_X1', 'NumHAcceptors_X1', 'TPSA_X1'])
feat_X1_df['col'] = df['col']
feat_X2_df = pd.DataFrame(df['features_X2'].tolist(), columns=['MolWt_X2', 'LogP_X2', 'NumHDonors_X2', 'NumHAcceptors_X2', 'TPSA_X2'])
feat_X2_df['col'] = df['col']

df = pd.merge(df, pd.merge(feat_X1_df, feat_X2_df,on='col'), on='col')
df = df.drop(['X1', 'X2', 'w', 'features_X1', 'features_X2'], axis=1)


print(f"Dataset ready (shape: {df.shape}), running on deep neural network now.")

model_full = DNN(df.iloc[:,3:],to_categorical(df["y"]), 5, 8000, 1)