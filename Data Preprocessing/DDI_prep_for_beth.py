import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from scipy import stats
import pyarrow.parquet as pq
import pyarrow as pa
import sqlalchemy
from sqlalchemy.engine import URL
import mysql.connector

#change to your path to the data, if different
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

#creating table of interaction codes
df_y_codes = pd.DataFrame(ddi['Y'].value_counts()).sort_values("count").reset_index()
df_y_codes = df_y_codes.head(20)
df_y_codes = df_y_codes.sort_values('Y').reset_index()
df_y_codes['int_reduced_code'] = df_y_codes.index
df_y_codes = df_y_codes.drop('count', axis=1)
df_y_codes['int_code'] = df_y_codes['Y']
df_y_codes.drop(['Y','index'], axis=1, inplace=True)

#listing longer explanations of interaction types for later use
interaction_types = ddi[['Y','Map']].drop_duplicates(subset=['Y'])
interaction_types.rename(columns={'Y': 'int_code', 'Map': 'int_desc'}, inplace=True)

# Database connection string

engine = sqlalchemy.create_engine('mysql+mysqlconnector://admin:group7@127.0.0.1:1234/DDI')

# Insert DataFrame into SQL table
df_y_codes.to_sql('y_codes_1', engine, if_exists='replace', index=False)
interaction_types.to_sql('interactions_1', engine, if_exists='replace', index=False)


db=mysql.connector.connect(host="127.0.0.1", port="1234", user="admin", password="group7",database="DDI")
cursor=db.cursor()

sql1 = "INSERT INTO interactions SELECT * FROM interactions_1"
sql2 = "INSERT INTO y_codes SELECT * FROM y_codes_1"
sql3 = "DROP TABLE y_codes_1"
sql4 = "DROP TABLE interactions_1"
cursor.execute(sql1)
cursor.execute(sql2)
cursor.execute(sql3)
cursor.execute(sql4)

db.close()

