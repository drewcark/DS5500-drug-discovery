import streamlit as st
st.set_page_config(page_title="DDI Predictor", layout="centered")

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.QED import qed
import mysql.connector
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

@st.cache_resource
def load_ffn_model():
    return load_model("dnn_model_red.keras")

model = load_ffn_model()

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2038)
    arr = np.zeros((2038,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

    desc = [
        mol.GetNumAtoms(),
        mol.GetNumHeavyAtoms(),
        mol.GetNumBonds(),
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        qed(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol)
    ]
    return np.concatenate([arr, desc])

def insert_to_db(sm1, sm2, result_code):
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1", port=3306,
            user="admin", password="group7", database="DDI"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inputs (molecule1, molecule2, result, result_reduced)
            VALUES (%s, %s, %s, %s)
        """, (sm1, sm2, result_code, True))
        conn.commit()
        conn.close()
    except Exception as e:
        pass

st.title("Drug Interaction Predictor")
st.markdown("Enter **two SMILES** to predict interaction type using the trained neural network.")

sm1 = st.text_input("SMILES for Drug 1")
sm2 = st.text_input("SMILES for Drug 2")

if st.button("Predict Interaction"):
    mol1 = Chem.MolFromSmiles(sm1)
    mol2 = Chem.MolFromSmiles(sm2)

    if not mol1 or not mol2:
        st.error("Invalid SMILES input.")
    else:
        f1 = extract_features(sm1)
        f2 = extract_features(sm2)

        if f1 is None or f2 is None:
            st.error("Feature extraction failed.")
        else:
            full_vector = np.concatenate([f1, f2]).reshape(1, -1)
            try:
                prediction = model.predict(full_vector)[0]
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                
                st.success(f"Predicted interaction class: **{predicted_class}**")

                insert_to_db(sm1, sm2, predicted_class)
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
                
                st.markdown("---")
st.subheader("All Interaction Types")

interaction_types_text = """
Class 0: Drug1 may increase the photosensitizing activities of Drug2.
Class 1: Drug1 may increase the ototoxic activities of Drug2.
Class 2: Drug1 may decrease the antiplatelet activities of Drug2.
Class 3: Drug1 may increase the dermatologic adverse activities of Drug2.
Class 4: The risk or severity of hypertension can be increased when #Drug2 is combined with Drug1.
Class 5: Drug1 may increase the vasodilatory activities of Drug2.
Class 6: #Drug1 may increase the hypotensive and central nervous system depressant (CNS depressant) activities of Drug2.
Class 7: The risk or severity of hyperkalemia can be increased when Drug1 is combined with Drug2.
Class 8: The protein binding of #Drug2 can be decreased when combined with #Drug1.
Class 9: #Drug1 may increase the central neurotoxic activities of #Drug2.
Class 10: #Drug1 may decrease effectiveness of #Drug2 as a diagnostic agent.
Class 11: #Drug1 may increase the bronchoconstrictory activities of #Drug2.
Class 12: The risk or severity of heart failure can be increased when #Drug2 is combined with #Drug1.
Class 13: #Drug1 may decrease the analgesic activities of #Drug2.
Class 14: The risk or severity of hypotension can be increased when #Drug1 is combined with #Drug2.
Class 15: The bioavailability of #Drug2 can be increased when combined with #Drug1.
Class 16: #Drug1 may increase the excretion rate of #Drug2 which could result in a lower serum level and potentially a reduction in efficacy.
Class 17: #Drug1 may increase the hyperglycemic activities of #Drug2.
Class 18: #Drug1 may increase the central nervous system depressant (CNS depressant) and hypertensive activities of #Drug2.
Class 19: The risk of a hypersensitivity reaction to #Drug2 is increased when it is combined with #Drug1.
"""

st.text_area("Interaction Class Reference", interaction_types_text.strip(), height=400)

            
