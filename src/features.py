from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys, Lipinski, Crippen, rdMolChemicalFeatures
from rdkit.Chem.Fragments import fr_halogen
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


def _chemical_structure_features(mol) -> Dict[str, float]:
    """
    Extract chemical structure features that correlate with melting point:
    - Hydrogen bonding descriptors
    - Symmetry measures  
    - Rotatable bonds
    - Molecular flexibility
    """
    if mol is None:
        return {
            "hbd_count": 0.0,
            "hba_count": 0.0, 
            "rotatable_bonds": 0.0,
            "heavy_atoms": 0.0,
            "rings": 0.0,
            "aromatic_rings": 0.0,
            "fused_rings": 0.0,
            "molecular_weight": 0.0,
            "logp": 0.0,
            "tpsa": 0.0,
            "balaban_j": 0.0,
            "kappa1": 0.0,
            "kappa2": 0.0,
            "kappa3": 0.0,
            "chi0v": 0.0,
            "chi1v": 0.0,
            "fraction_csp3": 0.0,
            "halogen_count": 0.0,
            "formal_charge": 0.0,
        }
    
    try:
        features = {}
        
        # Hydrogen bonding - critical for melting point
        features["hbd_count"] = Descriptors.NumHDonors(mol)
        features["hba_count"] = Descriptors.NumHAcceptors(mol)
        
        # Molecular flexibility
        features["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
        features["heavy_atoms"] = mol.GetNumHeavyAtoms()
        
        # Ring systems - affect crystal packing
        features["rings"] = Descriptors.RingCount(mol)
        features["aromatic_rings"] = Descriptors.NumAromaticRings(mol)
        features["fused_rings"] = Descriptors.NumAliphaticRings(mol)
        
        # Basic molecular properties
        features["molecular_weight"] = Descriptors.MolWt(mol)
        features["logp"] = Crippen.MolLogP(mol)
        features["tpsa"] = Descriptors.TPSA(mol)
        
        # Topological indices (symmetry measures)
        features["balaban_j"] = Descriptors.BalabanJ(mol)
        features["kappa1"] = Descriptors.Kappa1(mol)
        features["kappa2"] = Descriptors.Kappa2(mol)
        features["kappa3"] = Descriptors.Kappa3(mol)
        
        # Connectivity indices
        features["chi0v"] = Descriptors.Chi0v(mol)
        features["chi1v"] = Descriptors.Chi1v(mol)
        
        # Hybridization (sp3 carbons are more flexible)
        features["fraction_csp3"] = Descriptors.FractionCsp3(mol)
        
        # Halogen count (affects melting point significantly)
        features["halogen_count"] = fr_halogen(mol)
        
        # Formal charge
        features["formal_charge"] = Chem.rdmolops.GetFormalCharge(mol)
        
        return features
        
    except Exception:
        # Return zeros if calculation fails
        return {
            "hbd_count": 0.0,
            "hba_count": 0.0, 
            "rotatable_bonds": 0.0,
            "heavy_atoms": 0.0,
            "rings": 0.0,
            "aromatic_rings": 0.0,
            "fused_rings": 0.0,
            "molecular_weight": 0.0,
            "logp": 0.0,
            "tpsa": 0.0,
            "balaban_j": 0.0,
            "kappa1": 0.0,
            "kappa2": 0.0,
            "kappa3": 0.0,
            "chi0v": 0.0,
            "chi1v": 0.0,
            "fraction_csp3": 0.0,
            "halogen_count": 0.0,
            "formal_charge": 0.0,
        }


def build_chemical_structure_features(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    """Build chemical structure features from SMILES"""
    def to_mol(s: str):
        try:
            return Chem.MolFromSmiles(s)
        except Exception:
            return None
    
    rows: List[Dict[str, float]] = []
    for s in df[smiles_col].astype(str).tolist():
        mol = to_mol(s)
        rows.append(_chemical_structure_features(mol))
    
    feat = pd.DataFrame(rows)
    # Replace inf/nan that may arise from calculation errors
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat.add_prefix("CHEM_")


def build_advanced_features(df_chem: pd.DataFrame, use_polynomial: bool = False, use_interactions: bool = False) -> pd.DataFrame:
    """
    Build advanced features from chemical structure features:
    - Polynomial features for key descriptors
    - Meaningful chemical interactions
    - Derived ratios and indices
    """
    advanced_features = pd.DataFrame(index=df_chem.index)
    
    # Key chemical ratios that affect melting point
    eps = 1e-8  # Small constant to avoid division by zero
    
    # H-bonding capacity ratio
    hbd = df_chem.get('CHEM_hbd_count', 0) 
    hba = df_chem.get('CHEM_hba_count', 0)
    advanced_features['hbond_ratio'] = (hbd + eps) / (hba + eps)
    advanced_features['total_hbond'] = hbd + hba
    
    # Flexibility vs rigidity
    rot_bonds = df_chem.get('CHEM_rotatable_bonds', 0)
    rings = df_chem.get('CHEM_rings', 0)
    aromatic_rings = df_chem.get('CHEM_aromatic_rings', 0)
    advanced_features['flexibility_ratio'] = (rot_bonds + eps) / (rings + eps)
    advanced_features['rigidity_index'] = aromatic_rings + rings - rot_bonds
    
    # Size-normalized descriptors
    mw = df_chem.get('CHEM_molecular_weight', 1)
    heavy_atoms = df_chem.get('CHEM_heavy_atoms', 1)
    advanced_features['mw_per_atom'] = mw / (heavy_atoms + eps)
    advanced_features['tpsa_per_atom'] = df_chem.get('CHEM_tpsa', 0) / (heavy_atoms + eps)
    
    # Symmetry and compactness measures
    chi0v = df_chem.get('CHEM_chi0v', 0)
    chi1v = df_chem.get('CHEM_chi1v', 0)
    advanced_features['connectivity_ratio'] = (chi1v + eps) / (chi0v + eps)
    
    # Heteroatom and halogen effects
    halogen_count = df_chem.get('CHEM_halogen_count', 0)
    advanced_features['halogen_density'] = halogen_count / (heavy_atoms + eps)
    
    if use_polynomial:
        # Polynomial features for key melting point predictors
        key_features = ['CHEM_hbd_count', 'CHEM_hba_count', 'CHEM_molecular_weight', 
                       'CHEM_logp', 'CHEM_rings', 'CHEM_aromatic_rings']
        available_features = [f for f in key_features if f in df_chem.columns]
        
        if available_features:
            poly_data = df_chem[available_features].fillna(0)
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            poly_features = poly.fit_transform(poly_data)
            poly_names = [f"POLY_{name}" for name in poly.get_feature_names_out(available_features)]
            poly_df = pd.DataFrame(poly_features, index=df_chem.index, columns=poly_names)
            
            # Remove original features to avoid duplication
            original_cols = [col for col in poly_df.columns if not any(op in col for op in ['^2', ' '])]
            poly_df = poly_df.drop(columns=original_cols)
            
            advanced_features = pd.concat([advanced_features, poly_df], axis=1)
    
    if use_interactions:
        # Meaningful chemical interactions
        interactions = pd.DataFrame(index=df_chem.index)
        
        # H-bonding strength interaction
        interactions['hbond_strength'] = hbd * hba
        
        # Lipophilicity-size interaction
        logp = df_chem.get('CHEM_logp', 0)
        interactions['lipophilic_size'] = logp * mw
        
        # Ring system complexity
        interactions['ring_complexity'] = rings * aromatic_rings
        
        # Electrostatic-size interaction
        formal_charge = df_chem.get('CHEM_formal_charge', 0)
        interactions['charge_size'] = abs(formal_charge) * mw
        
        advanced_features = pd.concat([advanced_features, interactions], axis=1)
    
    # Remove low-variance features
    variance_threshold = VarianceThreshold(threshold=0.001)
    try:
        advanced_features_filtered = variance_threshold.fit_transform(advanced_features)
        feature_names = advanced_features.columns[variance_threshold.get_support()]
        advanced_features = pd.DataFrame(advanced_features_filtered, 
                                       index=advanced_features.index, 
                                       columns=feature_names)
    except:
        pass  # Keep original features if filtering fails
    
    return advanced_features.add_prefix("ADV_")


def get_group_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("Group ")]


def attach_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_smiles_basic: bool = False,
    use_smiles_tfidf: bool = False,
    use_chemical_structure: bool = False,
    use_advanced_features: bool = False,
    use_polynomial: bool = False,
    use_interactions: bool = False,
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

    if use_chemical_structure:
        tr_chem = build_chemical_structure_features(train)
        te_chem = build_chemical_structure_features(test)
        # Align columns just in case
        tr_chem, te_chem = tr_chem.align(te_chem, join="outer", axis=1, fill_value=0.0)
        train_feat = pd.concat([train_feat, tr_chem], axis=1)
        test_feat = pd.concat([test_feat, te_chem], axis=1)
        features.extend(list(tr_chem.columns))
        
        # Advanced features built from chemical structure features
        if use_advanced_features:
            tr_adv = build_advanced_features(tr_chem, use_polynomial=use_polynomial, use_interactions=use_interactions)
            te_adv = build_advanced_features(te_chem, use_polynomial=use_polynomial, use_interactions=use_interactions)
            # Align columns
            tr_adv, te_adv = tr_adv.align(te_adv, join="outer", axis=1, fill_value=0.0)
            train_feat = pd.concat([train_feat, tr_adv], axis=1)
            test_feat = pd.concat([test_feat, te_adv], axis=1)
            features.extend(list(tr_adv.columns))

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
