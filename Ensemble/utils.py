# =========================
# utils.py
# =========================

import pandas as pd

def load_probs(xlm_path, mde_path):
    xlm = pd.read_csv(xlm_path)
    mde = pd.read_csv(mde_path)
    return xlm, mde

def compute_ensemble(xlm, mde):
    ens = pd.DataFrame()
    ens["id"] = xlm["id"]
    ens["prob_0"] = (xlm["prob_0"] + mde["prob_0"]) / 2
    ens["prob_1"] = (xlm["prob_1"] + mde["prob_1"]) / 2
    ens["polarization"] = ens[["prob_0","prob_1"]].values.argmax(axis=1)
    return ens