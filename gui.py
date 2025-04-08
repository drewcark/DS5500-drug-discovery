import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import mysql.connector

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return None

def calculate_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def predict_interaction(similarity):
    if similarity > 0.8:
        return "High interaction risk (potentially synergistic or antagonistic)", 2, True
    elif similarity > 0.5:
        return "Moderate interaction risk (possible cross-reactivity)", 1, True
    else:
        return "Low interaction risk (likely safe together)", 0, False

def insert_input_to_db(sm1, sm2, result_code, reduced_flag):
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="admin",
            password="group7",
            database="DDI"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT IFNULL(MAX(inp_num), 0) + 1 FROM inputs")
        next_id = cursor.fetchone()[0]

        query = """
        INSERT INTO inputs (inp_num, molecule1, molecule2, result, result_reduced)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (next_id, sm1, sm2, result_code, reduced_flag))
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")

# ---------- Streamlit UI ----------
st.title("Drug Interaction Checker")
st.write("Enter two SMILES codes to predict interaction and save to database.")

smiles1 = st.text_input("SMILES for Drug 1", "")
smiles2 = st.text_input("SMILES for Drug 2", "")

if st.button("Check Interaction"):
    if smiles1 and smiles2:
        fp1 = get_fingerprint(smiles1)
        fp2 = get_fingerprint(smiles2)

        if fp1 and fp2:
            similarity = calculate_similarity(fp1, fp2)
            message, code, reduced = predict_interaction(similarity)

            st.subheader("Prediction")
            st.write(f"**Similarity Score:** {similarity:.2f}")
            st.write(f"**Result:** {message}")

            insert_input_to_db(smiles1, smiles2, code, reduced)
            st.success("✔️ Interaction saved to database.")
        else:
            st.error("❌ Invalid SMILES input.")
    else:
        st.warning(" Please enter both SMILES codes.")
