"""
sampling.py - SMOTE wrapper used only on the training split.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE, SMOTE_K_NEIGHBORS, SMOTE_STRATEGY


def resample_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = SMOTE_STRATEGY,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(y_train, return_counts=True)
    before = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"SMOTE before: {before}")

    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    unique_res, counts_res = np.unique(y_res, return_counts=True)
    after = {int(k): int(v) for k, v in zip(unique_res, counts_res)}
    print(f"SMOTE after:  {after}")
    assert len(X_res) >= len(X_train)
    return X_res, y_res
