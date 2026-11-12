# Lekgotla-Cyber-Shield

Open-methodology forensic framework for ransomware UTXO attribution and SADC
statutory admissibility.

## Reproducing the empirical results

The current validation script is **honest_validation_v3.py** (leakage-free).
See CHANGELOG.md for the correction history from v2.

### Option A — Docker (recommended)

    docker build -t lekgotla-validation .
    docker run --rm lekgotla-validation

### Option B — Local Python

    pip install -r requirements.txt
    python honest_validation_v3.py

### Expected output

    Frozen: eps=0.7, min_samples=20, validation F1=0.4376
    Test mean: P=0.2833 (SD 0.0069), R=0.9775 (SD 0.0305), F1=0.4393 (SD 0.0110)
    F1 range: 0.4138 - 0.4494

Environment: Python 3.10, numpy==1.24.3, scikit-learn==1.3.0. MIT License.
All data are synthetic; no real victims, addresses, or investigations.
