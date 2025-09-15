# run.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from Dataloader.Dataloader import load_data
from model import build_all_views, attribute_scores_single_k, class_scores


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-Max normalization"""
    min_val, max_val = np.min(x), np.max(x)
    if abs(max_val - min_val) < 1e-12:
        return np.ones_like(x) * 0.5
    return (x - min_val) / (max_val - min_val)


def main():
    # Load dataset
    data_path = "data/Scene_3v_4485n_15c_1_258.mat"
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    v1, v2, v3, labels = load_data(data_path)
    X_views = [v1, v2, v3]
    labels = labels.astype(int).flatten()

    # Precompute view structures
    views = build_all_views(X_views)

    #class outlier
    k_GB = list(range(5, 75, 10))
    lambdas = [round(x, 1) for x in np.linspace(0.1, 0.9, 9)]

    best = {"auc": -1.0, "k_gb": None, "lambda": None,
            "auc_cls": None, "auc_attr": None}
    results = []

    # Class outlier scores (fixed k_sample)
    scores_cls = class_scores(views)
    scores_cls = minmax_normalize(scores_cls)
    try:
        auc_cls_single = roc_auc_score(labels, scores_cls)
        print(f"[Info] Class-only AUC): {auc_cls_single:.4f}")
    except Exception as e:
        print(f"[Warn] Failed AUC(class-only):, err={e}")
        auc_cls_single = -1.0

    # Attribute outlier scores
    attr_results = {}
    for k_gb in k_GB:
        scores_attr = attribute_scores_single_k(views, k_gb)
        scores_attr = minmax_normalize(scores_attr)
        try:
            auc_attr_single = roc_auc_score(labels, scores_attr)
            print(f"[Info] Attribute-only AUC (k_gb={k_gb}): {auc_attr_single:.4f}")
        except Exception as e:
            print(f"[Warn] Failed AUC(attribute-only): k_gb={k_gb}, err={e}")
            auc_attr_single = -1.0
        attr_results[k_gb] = (scores_attr, auc_attr_single)

    # Fusion scores
    for k_gb, (scores_attr, auc_attr_single) in attr_results.items():
        for lam in lambdas:
            scores = (1.0 - lam) * scores_attr + lam * scores_cls
            try:
                auc = roc_auc_score(labels, scores)
            except Exception:
                auc = -1.0

            results.append({
                "k_gb": k_gb,
                "lambda": lam,
                "auc": auc,
                "auc_cls": auc_cls_single,
                "auc_attr": auc_attr_single
            })

            if auc > best["auc"]:
                best.update({
                    "auc": auc,
                    "k_gb": k_gb,
                    "lambda": lam,
                    "auc_cls": auc_cls_single,
                    "auc_attr": auc_attr_single
                })

    # Save results
    os.makedirs("result_test", exist_ok=True)
    out_path = os.path.join("result_test", f"{dataset_name}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_gb", "lambda", "auc", "auc_cls", "auc_attr"])
        writer.writerow([best["k_gb"], best["lambda"],
                         best["auc"], best["auc_cls"], best["auc_attr"]])
        for r in results:
            writer.writerow([r["k_gb"], r["lambda"],
                             r["auc"], r["auc_cls"], r["auc_attr"]])

    # Print best result
    print("\n=== Grid Search Result ===")
    if best["auc"] > 0:
        print(f"Best AUC = {best['auc']:.4f} "
              f"at k_gb = {best['k_gb']}, "
              f"lambda = {best['lambda']:.1f}")
        print(f"  ↳ Class-only AUC = {best['auc_cls']:.4f}")
        print(f"  ↳ Attribute-only AUC = {best['auc_attr']:.4f}")
    else:
        print("No valid AUC found (labels constant or no score differences).")


if __name__ == "__main__":
    main()
