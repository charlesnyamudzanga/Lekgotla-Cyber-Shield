# Changelog

## v3 (current)
- **Fixed critical label leakage**: v2's cluster-to-prediction mapping used
  ground-truth labels (`y_true`) at prediction time. v3 replaces this with a
  label-free operational triage rule: every node in any dense DBSCAN cluster
  is flagged; noise points are treated as background. Labels are used only
  for scoring.
- Added standard scaling of features (consistent with the manuscript).
- Documented five sub-populations implementing deliberate class overlap
  (32 core attacker, 8 mixer-obfuscated, 88 dense benign services,
  64 diffuse benign background, 8 corporate wallets).
- All manuscript metrics regenerated with v3:
  frozen params eps=0.7, min_samples=20;
  test mean F1=0.4393 (SD 0.0110), recall=0.9775, precision=0.2833.

## v2 (retained for transparency — DO NOT USE)
- Contained a label-mapping error identified during pre-submission audit.
- Its outputs do not correspond to a deployable, leakage-free pipeline.
- Kept in the repository so reviewers can verify the correction history.
