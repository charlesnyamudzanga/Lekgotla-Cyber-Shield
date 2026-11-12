#!/usr/bin/env python3
"""
honest_validation_v3.py — Lekgotla Cyber Shield empirical baseline (leakage-free).

CHANGELOG v2 -> v3:
  * FIXED: cluster-to-prediction mapping no longer uses ground-truth labels.
    v2 mapped clusters using y_true (label leakage). v3 flags ALL nodes in any
    dense DBSCAN cluster (label-free operational triage rule); noise = background.
  * ADDED: standard scaling of features (as described in the manuscript).
  * UPDATED: five documented sub-populations implementing deliberate class overlap:
      32 core attacker | 8 mixer-obfuscated attacker (20% of attackers)
      88 dense benign services | 64 diffuse background benign | 8 corporate wallets (5% of benign)
  * Protocol unchanged: 54-point grid on validation seeds 1000-1009; freeze; single
    test pass on seeds 0-9.

Expected output (numpy 1.24.3, scikit-learn 1.3.0, Python 3.10):
  Best validation: eps=0.7, min_samples=20, mean F1=0.4376
  Test: mean P=0.2833 (SD 0.0069), R=0.9775 (SD 0.0305), F1=0.4393 (SD 0.0110)
"""
import itertools
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

EPS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MS_GRID = [3, 5, 8, 10, 12, 15, 20, 25, 30]   # 6 x 9 = 54 combinations
VALIDATION_SEEDS = range(1000, 1010)
TEST_SEEDS = range(0, 10)

def generate_scenario(seed):
    """One 200-node synthetic ransomware incident graph; 3 topological features."""
    rng = np.random.RandomState(seed)
    core   = rng.normal([0.80, 0.80, 0.80], [0.07]*3, (32, 3))   # core attacker
    mixer  = rng.normal([0.30, 0.30, 0.60], [0.09]*3, (8, 3))    # obfuscated attacker
    b_serv = rng.normal([0.30, 0.30, 0.60], [0.06]*3, (88, 3))   # dense benign services
    b_bg   = rng.normal([0.45, 0.55, 0.25], [0.16]*3, (64, 3))   # diffuse benign background
    b_corp = rng.normal([0.80, 0.80, 0.80], [0.07]*3, (8, 3))    # corporate wallets (benign)
    X = np.vstack((core, mixer, b_serv, b_bg, b_corp))
    y = np.concatenate((np.ones(40), np.zeros(160)))             # labels used ONLY for scoring
    return StandardScaler().fit_transform(X), y

def evaluate(Xs, y, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs).labels_
    flagged = labels != -1          # label-free triage: all clustered nodes flagged
    tp = np.sum(flagged & (y == 1)); fp = np.sum(flagged & (y == 0))
    fn = np.sum(~flagged & (y == 1))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def main():
    val = {s: generate_scenario(s) for s in VALIDATION_SEEDS}
    print("Validation grid search (seeds 1000-1009)...")
    best_f1, best = -1.0, None
    for eps, ms in itertools.product(EPS_GRID, MS_GRID):
        m = np.mean([evaluate(*val[s], eps, ms)[2] for s in VALIDATION_SEEDS])
        if m > best_f1:
            best_f1, best = m, (eps, ms)
    print(f"Frozen: eps={best[0]}, min_samples={best[1]}, validation F1={best_f1:.4f}")

    print("\nSingle frozen test pass (seeds 0-9):")
    rows = []
    for s in TEST_SEEDS:
        p, r, f = evaluate(*generate_scenario(s), *best)
        rows.append((p, r, f))
        print(f"Seed {s}: P={p:.4f} R={r:.4f} F1={f:.4f}")
    a = np.array(rows)
    print(f"\nMean P={a[:,0].mean():.4f} (SD {a[:,0].std():.4f}) "
          f"R={a[:,1].mean():.4f} (SD {a[:,1].std():.4f}) "
          f"F1={a[:,2].mean():.4f} (SD {a[:,2].std():.4f})")
    print(f"F1 range: {a[:,2].min():.4f} - {a[:,2].max():.4f}")

if __name__ == "__main__":
    main()
