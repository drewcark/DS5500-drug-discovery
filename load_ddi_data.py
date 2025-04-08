import pandas as pd
import numpy as np
import sqlalchemy
import mysql.connector

ddi_fp = "/Users/bethfarr/Downloads/drugbank.csv"
ddi = pd.read_csv(ddi_fp, sep='\t')

ddi["Y"] = ddi["Y"].astype("category")
ddi["Map"] = ddi["Map"].astype("category")

interaction_counts = pd.DataFrame(ddi['Y'].value_counts().rename_axis('y').reset_index(name='count')).sort_values(by='count', ascending=False)
interaction_counts['row_num'] = interaction_counts.index + 1
interaction_counts['log_count'] = np.log(interaction_counts['count'])

df_y_codes = pd.DataFrame(ddi['Y'].value_counts()).sort_values("count").reset_index()
df_y_codes = df_y_codes.head(20)
df_y_codes = df_y_codes.sort_values('Y').reset_index()
df_y_codes['int_reduced_code'] = df_y_codes.index
df_y_codes = df_y_codes.drop('count', axis=1)
df_y_codes['int_code'] = df_y_codes['Y']
df_y_codes.drop(['Y','index'], axis=1, inplace=True)

interaction_types = ddi[['Y','Map']].drop_duplicates(subset=['Y'])
interaction_types.rename(columns={'Y': 'int_code', 'Map': 'int_desc'}, inplace=True)

engine = sqlalchemy.create_engine('mysql+mysqlconnector://admin:group7@127.0.0.1:3306/DDI')

df_y_codes.to_sql('y_codes_1', engine, if_exists='replace', index=False)
interaction_types.to_sql('interactions_1', engine, if_exists='replace', index=False)

db = mysql.connector.connect(host="127.0.0.1", port="3306", user="admin", password="group7", database="DDI")
cursor = db.cursor()
cursor.execute("INSERT IGNORE INTO interactions SELECT * FROM interactions_1")
cursor.execute("INSERT IGNORE INTO y_codes SELECT * FROM y_codes_1")
cursor.execute("DROP TABLE y_codes_1")
cursor.execute("DROP TABLE interactions_1")
db.commit()

kaggle_fp = "/Users/bethfarr/Downloads/chembl_22_clean_1576904_sorted_std_final.csv"
smiles = pd.read_csv(kaggle_fp, sep='\t', header=None, names=["smiles", "name"])

np.random.seed(42)
smiles["result"] = np.random.randint(0, len(df_y_codes), size=len(smiles))
smiles["result_reduced"] = smiles["result"] < 10
smiles["inp_num"] = range(1, len(smiles) + 1)
smiles.rename(columns={"smiles": "molecule1"}, inplace=True)
smiles["molecule2"] = smiles["molecule1"]

inputs_df = smiles[["inp_num", "molecule1", "molecule2", "result", "result_reduced"]]
inputs_df.to_sql('inputs', engine, if_exists='append', index=False)

cursor.close()
db.close()

print("Data loaded successfully into MySQL!")
