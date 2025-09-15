# model.py
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from GB_generation import GB_Gen
from sklearn.metrics import pairwise_distances


def _row_normalize(mat: np.ndarray, uniform_when_zero: bool = True) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    if uniform_when_zero:
        zero_rows = (s <= 1e-12).flatten()
        if np.any(zero_rows):
            mat[zero_rows, :] = 1.0 / mat.shape[1]
            s[zero_rows, 0] = 1.0
    else:
        s[s <= 1e-12] = 1.0
    return mat / s


def _soft_membership(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Soft membership matrix via heat kernel"""
    D = cdist(X, centers)
    sigma = float(np.median(D))
    sigma = max(sigma, 1e-12)
    R = np.exp(-(D ** 2) / (2.0 * sigma * sigma))

    return R


def _one_hot_membership(point_to_gb: np.ndarray, n_gb: int) -> np.ndarray:
    n = point_to_gb.shape[0]
    M = np.zeros((n, n_gb), dtype=np.int32)
    M[np.arange(n), point_to_gb] = 1
    return M


def _soft_bipartite_similarity(Rv: np.ndarray, Ru: np.ndarray, k: int = 50) -> np.ndarray:
    def sparsify_R(R: np.ndarray, k: int) -> np.ndarray:
        n, g = R.shape
        R_sparse = np.zeros_like(R)
        for p in range(g):
            col = R[:, p].copy()
            top_idx = np.argpartition(-col, min(k, n-1))[:k]
            R_sparse[top_idx, p] = col[top_idx]
            col_sum = R_sparse[:, p].sum()
            if col_sum > 1e-12:
                R_sparse[:, p] /= col_sum
        return R_sparse

    Rv_sparse = sparsify_R(Rv, k)
    Ru_sparse = sparsify_R(Ru, k)
    n = Rv.shape[0]
    S = (Rv_sparse.T @ Ru_sparse) / max(n, 1)
    avg_nonzero_v = np.mean([np.count_nonzero(Rv_sparse[:, p]) for p in range(Rv_sparse.shape[1])])
    avg_nonzero_u = np.mean([np.count_nonzero(Ru_sparse[:, p]) for p in range(Ru_sparse.shape[1])])
    print(f"[Debug] Rv nonzero ≈ {avg_nonzero_v:.2f}, Ru ≈ {avg_nonzero_u:.2f} (target ≈ k)")
    return S


def _jaccard_between_views(Mv: np.ndarray, Mu: np.ndarray) -> np.ndarray:
    inter = (Mv.T @ Mu).astype(float)
    size_v = Mv.sum(axis=0).reshape(-1, 1)
    size_u = Mu.sum(axis=0).reshape(1, -1)
    union = size_v + size_u - inter
    S = inter / np.maximum(union, 1e-12)
    return S


def _multi_step_Z(P_vu: np.ndarray, P_uv: np.ndarray, t: int = 3) -> np.ndarray:
    if t <= 1:
        return P_vu.copy()
    B_u = P_uv @ P_vu
    Z = P_vu @ np.linalg.matrix_power(B_u, t - 1)
    return Z


def _trajectory_Z_cosine(P_vu: np.ndarray, P_uv: np.ndarray, t: int = 3) -> np.ndarray:
    g_u = P_uv.shape[0]
    B_u = P_uv @ P_vu
    blocks_A, blocks_B = [], []
    A_k = P_vu.copy()
    B_pow = np.eye(g_u)
    blocks_A.append(A_k)
    blocks_B.append(B_pow)
    for _ in range(1, t):
        B_pow = B_pow @ B_u
        A_k = A_k @ B_u
        blocks_A.append(A_k)
        blocks_B.append(B_pow)
    A_concat = np.concatenate(blocks_A, axis=1)
    B_concat = np.concatenate(blocks_B, axis=1)
    A_norm = np.maximum(np.linalg.norm(A_concat, axis=1, keepdims=True), 1e-12)
    B_norm = np.maximum(np.linalg.norm(B_concat, axis=1, keepdims=True), 1e-12)
    Z = (A_concat @ B_concat.T) / (A_norm @ B_norm.T)
    Z = np.clip(Z, 0.0, 1.0)
    return Z


def build_view_structures(X: np.ndarray) -> dict:
    """Build structures for one view"""
    n, d = X.shape
    gb_list = GB_Gen(X)
    g = len(gb_list)
    centers = np.stack([gb.mean(axis=0) if len(gb) > 0 else np.zeros(d) for gb in gb_list], axis=0)
    D_pc = cdist(X, centers)
    point_to_gb = D_pc.argmin(axis=1).astype(int)
    M = _one_hot_membership(point_to_gb, g)
    R = _soft_membership(X, centers)
    Cdist = cdist(centers, centers)
    neighbor_order = np.argsort(Cdist, axis=1)
    return dict(
        X=X,
        gb_list=gb_list,
        centers=centers,
        point_to_gb=point_to_gb,
        M=M,
        R=R,
        Cdist=Cdist,
        neighbor_order=neighbor_order
    )


def build_all_views(view_list):
    return [build_view_structures(Xv) for Xv in view_list]


def attribute_scores_single_k(views, k: int) -> np.ndarray:
    """Attribute outlier score (granular-ball version)"""
    V = len(views)
    n = views[0]['X'].shape[0]
    centers_list = [vw['centers'] for vw in views]
    order_list = [vw['neighbor_order'] for vw in views]
    assign_list = [vw['point_to_gb'] for vw in views]
    X_list = [vw['X'] for vw in views]
    per_view_scores = []
    for v in range(V):
        centers = centers_list[v]
        g_v, d_v = centers.shape
        order = order_list[v]
        p2g = assign_list[v]
        Xv = X_list[v]
        if g_v <= 1:
            updated_centers = centers.copy()
        else:
            k_eff = int(min(max(k, 1), max(g_v - 1, 1)))
            neigh_idx = order[:, 1:k_eff + 1]
            updated_centers = centers[neigh_idx].mean(axis=1)
        upd_center_i = updated_centers[p2g]
        sqdist = np.sum((Xv - upd_center_i) ** 2, axis=1)
        min_val, max_val = np.min(sqdist), np.max(sqdist)
        if abs(max_val - min_val) < 1e-12:
            normed = np.ones_like(sqdist) * 0.5
        else:
            normed = (sqdist - min_val) / (max_val - min_val)
        per_view_scores.append(normed)
    scores_attr = np.max(np.stack(per_view_scores, axis=1), axis=1)
    return scores_attr





def class_scores(views, sparse: int = 105) -> np.ndarray:
    """Class outlier score (sparse fingerprint)"""
    V = len(views)
    n = views[0]['X'].shape[0]
    R_list = [vw['R'] for vw in views]
    assign_list = [vw['point_to_gb'] for vw in views]
    R_sparse_list = []
    for R in R_list:
        n, g = R.shape
        R_sparse = np.zeros_like(R)
        for p in range(g):
            col = R[:, p].copy()
            top_idx = np.argpartition(-col, sparse)[:sparse]
            R_sparse[top_idx, p] = col[top_idx]
            col_sum = R_sparse[:, p].sum()
            if col_sum > 1e-12:
                R_sparse[:, p] /= col_sum
        R_sparse_list.append(R_sparse)
    avg_nonzero = np.mean([np.count_nonzero(R_sparse[:, p])
                           for R_sparse in R_sparse_list
                           for p in range(R_sparse.shape[1])])

    scores_cls = np.zeros(n, dtype=float)
    for i in range(n):
        total_diff = 0.0
        p_idx = [assign_list[v][i] for v in range(V)]
        for v in range(V):
            finger_v = R_sparse_list[v][:, p_idx[v]].astype(float)
            for u in range(v + 1, V):
                finger_u = R_sparse_list[u][:, p_idx[u]].astype(float)
                diff = finger_v - finger_u
                d = np.dot(diff, diff)
                total_diff += d
        scores_cls[i] = total_diff
    return scores_cls
