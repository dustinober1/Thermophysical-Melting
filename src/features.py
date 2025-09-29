from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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

    return train_feat, test_feat, features
