# Network IDS Project (Beginner Friendly)

A simple malware/threat analyzer for network logs using machine learning.

## Project Goal
Classify network records as `malicious` or `benign`.

## Quick Start
1. Create virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install --default-timeout 120 --retries 10 -r requirements.txt
   ```
3. Download a real public dataset (KDD Cup 99):
   ```powershell
   python scripts/download_dataset.py
   ```
4. Run end-to-end pipeline:
   ```powershell
   python scripts/run_pipeline.py
   ```
5. Run demo prediction:
   ```powershell
   python scripts/run_demo.py
   ```

## Dataset Source
- `KDD Cup 99` fetched via `scikit-learn` (`fetch_kddcup99`)
- Original source: UCI KDD Archive (public/free)

## Data Format
Expected columns:
- `duration`
- `protocol`
- `src_bytes`
- `dst_bytes`
- `flag`
- `label` (values: `benign` or `malicious`)

## Output
- Trained model: `models/best_model.pkl`
- Metrics: `reports/metrics.json`
- Model comparison (CV): `reports/model_comparison.json`
- Confusion matrix image: `reports/confusion_matrix.png` (optional, only if `matplotlib` + `seaborn` installed)

## What the pipeline now does
- Adds engineered features (`bytes_ratio`, `is_zero_traffic`)
- Tries multiple models (Random Forest, Decision Tree, Logistic Regression)
- Uses 5-fold cross-validation and selects the best model by weighted F1
