from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys
from mordred import Calculator, descriptors


HALOGENS = ["F", "Cl", "Br", "I"]
HEAVY_ATOMS = ["C", "N", "O", "S", "P"] + HALOGENS


def _count_substring(s: str, sub: str) -> int:
    return s.count(sub)


def _max_parentheses_depth(s: str) -> int:
    depth = max_depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return max_depth


def _basic_smiles_stats(smiles: str) -> Dict[str, float]:
    s = smiles or ""
    L = len(s)
    features: Dict[str, float] = {
        "smiles_len": L,
        "num_equals": s.count("="),
        "num_hash": s.count("#"),
        "num_paren_open": s.count("("),
        "num_paren_close": s.count(")"),
        "paren_depth_max": _max_parentheses_depth(s),
        "num_brackets_open": s.count("["),
        "num_brackets_close": s.count("]"),
        "num_plus": s.count("+"),
        "num_minus": s.count("-"),
        "num_dots": s.count("."),
        "num_ring_digits": sum(ch.isdigit() for ch in s),
    }

    # Atom symbol counts (simple heuristics; counts 'Cl' and 'Br' before single-letter atoms)
    for X in HALOGENS:
        features[f"atom_{X}"] = _count_substring(s, X)
    # Avoid double counting 'Cl'/'Br' when counting 'C'/'B': pre-remove those tokens
    reduced = s.replace("Cl", "").replace("Br", "")
    for X in ["C", "N", "O", "S", "P", "B", "H", "I", "F"]:
        features[f"atom_{X}_single"] = _count_substring(reduced, X)

    # Aromatic lowercase atoms
    for x in ["c", "n", "o", "s", "p"]:
        features[f"aromatic_{x}"] = s.count(x)

    features["num_hetero"] = (
        features["atom_N_single"]
        + features["atom_O_single"]
        + features["atom_S_single"]
        + features["atom_P_single"]
        + features["atom_F"]
        + features["atom_Cl"]
        + features["atom_Br"]
        + features["atom_I"]
    )
    features["frac_hetero"] = features["num_hetero"] / (L + 1e-6)
    features["num_aromatic"] = sum(features[k] for k in features if k.startswith("aromatic_"))
    features["frac_aromatic"] = features["num_aromatic"] / (L + 1e-6)

    return features


def build_basic_smiles_features(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for s in df[smiles_col].astype(str).tolist():
        rows.append(_basic_smiles_stats(s))
    feat = pd.DataFrame(rows)
    # Replace inf/nan that may arise from empty strings
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat.add_prefix("S_")


def get_group_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("Group ")]


def attach_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_smiles_basic: bool = False,
    use_smiles_tfidf: bool = False,
    tfidf_ngram_min: int = 2,
    tfidf_ngram_max: int = 5,
    tfidf_min_df: int = 2,
    svd_components: int = 256,
    random_state: int = 42,
    use_rdkit_desc: bool = False,
    use_morgan: bool = False,
    morgan_radius: int = 2,
    morgan_nbits: int = 1024,
    use_maccs: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    features: List[str] = []

    # Group features
    group_cols = get_group_columns(train)
    features.extend(group_cols)

    train_feat = train[group_cols].copy()
    test_feat = test[group_cols].copy()

    if use_smiles_basic:
        tr_s = build_basic_smiles_features(train)
        te_s = build_basic_smiles_features(test)
        # Align columns just in case
        tr_s, te_s = tr_s.align(te_s, join="outer", axis=1, fill_value=0.0)
        train_feat = pd.concat([train_feat, tr_s], axis=1)
        test_feat = pd.concat([test_feat, te_s], axis=1)
        features.extend(list(tr_s.columns))

    if use_smiles_tfidf:
        # Character-level TF-IDF on SMILES, then SVD to dense low-dim
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(tfidf_ngram_min, tfidf_ngram_max),
            min_df=tfidf_min_df,
        )
        tr_texts = train["SMILES"].astype(str).tolist()
        te_texts = test["SMILES"].astype(str).tolist()
        Xtr_sparse = vec.fit_transform(tr_texts)
        Xte_sparse = vec.transform(te_texts)

        svd = TruncatedSVD(n_components=min(svd_components, Xtr_sparse.shape[1]-1), random_state=random_state)
        Xtr_svd = svd.fit_transform(Xtr_sparse)
        Xte_svd = svd.transform(Xte_sparse)

        svd_cols = [f"TFIDF_SVD_{i:03d}" for i in range(Xtr_svd.shape[1])]
        tr_df = pd.DataFrame(Xtr_svd, index=train.index, columns=svd_cols)
        te_df = pd.DataFrame(Xte_svd, index=test.index, columns=svd_cols)
        train_feat = pd.concat([train_feat, tr_df], axis=1)
        test_feat = pd.concat([test_feat, te_df], axis=1)
        features.extend(svd_cols)

    if use_rdkit_desc or use_morgan or use_maccs:
        # Convert SMILES to mols
        def to_mol(s: str):
            try:
                return Chem.MolFromSmiles(s)
            except Exception:
                return None

        tr_smiles = train["SMILES"].astype(str).tolist()
        te_smiles = test["SMILES"].astype(str).tolist()
        tr_mols = [to_mol(s) for s in tr_smiles]
        te_mols = [to_mol(s) for s in te_smiles]

        if use_rdkit_desc:
            # Mordred descriptors (2D only)
            calc = Calculator(descriptors, ignore_3D=True)
            tr_desc = calc.pandas(tr_mols)
            te_desc = calc.pandas(te_mols)
            # Clean
            tr_desc = tr_desc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            te_desc = te_desc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Align columns
            tr_desc, te_desc = tr_desc.align(te_desc, join="outer", axis=1, fill_value=0.0)
            # Columns to strings
            tr_desc.columns = tr_desc.columns.astype(str)
            te_desc.columns = te_desc.columns.astype(str)
            tr_desc = tr_desc.add_prefix("RD_")
            te_desc = te_desc.add_prefix("RD_")
            train_feat = pd.concat([train_feat, tr_desc], axis=1)
            test_feat = pd.concat([test_feat, te_desc], axis=1)
            features.extend(list(tr_desc.columns))

        if use_morgan:
            def morgan_fp(mol):
                if mol is None:
                    return np.zeros(morgan_nbits, dtype=np.int8)
                bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_nbits)
                arr = np.zeros((morgan_nbits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                return arr

            tr_fp = np.vstack([morgan_fp(m) for m in tr_mols])
            te_fp = np.vstack([morgan_fp(m) for m in te_mols])
            cols = [f"MG_{morgan_radius}_{morgan_nbits}_{i:04d}" for i in range(tr_fp.shape[1])]
            tr_df = pd.DataFrame(tr_fp, index=train.index, columns=cols)
            te_df = pd.DataFrame(te_fp, index=test.index, columns=cols)
            train_feat = pd.concat([train_feat, tr_df], axis=1)
            test_feat = pd.concat([test_feat, te_df], axis=1)
            features.extend(cols)

        if use_maccs:
            def maccs_fp(mol):
                if mol is None:
                    return np.zeros(167, dtype=np.int8)
                keys = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((167,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(keys, arr)
                return arr

            tr_fp = np.vstack([maccs_fp(m) for m in tr_mols])
            te_fp = np.vstack([maccs_fp(m) for m in te_mols])
            cols = [f"MACCS_{i:03d}" for i in range(tr_fp.shape[1])]
            tr_df = pd.DataFrame(tr_fp, index=train.index, columns=cols)
            te_df = pd.DataFrame(te_fp, index=test.index, columns=cols)
            train_feat = pd.concat([train_feat, tr_df], axis=1)
            test_feat = pd.concat([test_feat, te_df], axis=1)
            features.extend(cols)

    return train_feat, test_feat, features
