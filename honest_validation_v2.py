import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. Synthetic Scenario Generation (Section 4.3) ---
def generate_scenario(seed, num_nodes=200):
    np.random.seed(seed)
    
    # 3 Features: Transaction volume, address reuse rate, clustering coefficient
    # Core attacker (normal distribution)
    core_attacker = np.random.normal(loc=[0.2, 0.8, 0.7], scale=[0.1, 0.1, 0.1], size=(int(num_nodes*0.15), 3))
    
    # Mixer-obfuscated attacker (overlaps with benign in volume)
    mixer_attacker = np.random.normal(loc=[0.5, 0.6, 0.5], scale=[0.15, 0.15, 0.15], size=(int(num_nodes*0.05), 3))
    
    # Benign (mostly normal, 5% high-volume overlaps with mixer)
    benign = np.random.normal(loc=[0.5, 0.2, 0.3], scale=[0.15, 0.1, 0.1], size=(int(num_nodes*0.80), 3))
    
    X = np.vstack((core_attacker, mixer_attacker, benign))
    # 1 = Attacker (Core + Mixer), 0 = Benign
    y_true = np.concatenate((np.ones(len(core_attacker)), np.ones(len(mixer_attacker)), np.zeros(len(benign))))
    
    return X, y_true

# --- 2. Evaluation Function ---
def evaluate_clustering(X, y_true, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    
    # Map clusters to predictions (simplified: if cluster contains >50% attackers, predict 1)
    y_pred = np.zeros_like(y_true)
    for cluster_id in set(labels):
        if cluster_id == -1: # Noise
            continue
        cluster_indices = np.where(labels == cluster_id)[0]
        if np.mean(y_true[cluster_indices]) > 0.5:
            y_pred[cluster_indices] = 1
            
    return precision_score(y_true, y_pred, zero_division=0), \
           recall_score(y_true, y_pred, zero_division=0), \
           f1_score(y_true, y_pred, zero_division=0)

# --- 3. Grid Search on Validation Seeds (Section 4.4) ---
print("Starting Validation Grid Search (Seeds 1000-1009)...")
validation_seeds = range(1000, 1010)
eps_range = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
min_samples_range = [3, 5, 8, 10, 12, 15, 20, 25, 30] # 6 * 9 = 54 combinations

best_f1 = 0
best_params = (0, 0)

for eps in eps_range:
    for min_samples in min_samples_range:
        f1_scores = []
        for seed in validation_seeds:
            X, y_true = generate_scenario(seed)
            _, _, f1 = evaluate_clustering(X, y_true, eps, min_samples)
            f1_scores.append(f1)
            
        mean_f1 = np.mean(f1_scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = (eps, min_samples)

print(f"Best Validation F1: {best_f1:.4f} with params eps={best_params[0]}, min_samples={best_params[1]}")
print(f"Parameters Frozen. Starting Test Evaluation...\n")

# --- 4. Frozen Test Evaluation (Section 5.1.2) ---
test_seeds = range(0, 10)
test_results = []

for seed in test_seeds:
    X, y_true = generate_scenario(seed)
    prec, rec, f1 = evaluate_clustering(X, y_true, best_params[0], best_params[1])
    test_results.append((seed, prec, rec, f1))
    print(f"Seed {seed}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

# Summary
f1_values = [res[3] for res in test_results]
print(f"\nSummary: F1 Mean = {np.mean(f1_values):.4f}, SD = {np.std(f1_values):.4f}")
