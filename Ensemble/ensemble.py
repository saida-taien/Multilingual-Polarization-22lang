# =========================
# ensemble.py (for 22 lang)
# =========================

import os
import pandas as pd
from google.colab import files
from utils import load_probs, compute_ensemble

languages = [
    "amh","arb","ben","deu","eng","fas","hau","hin","ita","khm",
    "mya","nep","ori","pan","pol","rus","spa","swa","tel","tur",
    "urd","zho"
]

INPUT_FOLDER = "./probabilities"

for lang in languages:
    xlm_path = os.path.join(INPUT_FOLDER, f"pred_{lang}_xlm_probs.csv")
    mde_path = os.path.join(INPUT_FOLDER, f"pred_{lang}_mdeberta_probs.csv")

    if not os.path.exists(xlm_path) or not os.path.exists(mde_path):
        print(f"Skipping {lang} (missing probability files)")
        continue

    xlm, mde = load_probs(xlm_path, mde_path)
    ens = compute_ensemble(xlm, mde)

    final_file = f"pred_{lang}.csv"
    ens[["id", "polarization"]].to_csv(final_file, index=False)

    files.download(final_file)
    print(f"Downloaded: {final_file}")

print("All available languages processed and downloaded.")