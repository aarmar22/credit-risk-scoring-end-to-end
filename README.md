https://github.com/aarmar22/credit-risk-scoring-end-to-end/releases

# End-to-End Credit Risk Scoring: PyTorch, Tableau, Features 🚦

[![Releases](https://img.shields.io/github/v/release/aarmar22/credit-risk-scoring-end-to-end)](https://github.com/aarmar22/credit-risk-scoring-end-to-end/releases)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.7%2B-orange)](https://pytorch.org/) [![License](https://img.shields.io/github/license/aarmar22/credit-risk-scoring-end-to-end)](LICENSE)

Loan grading model built with public Kaggle data. The repo shows a full workflow: data cleaning, feature engineering, neural network training with PyTorch, model evaluation, and a Tableau dashboard that compares predictions with actual loan grades.

Quick visual (sample dashboard):
![Tableau Dashboard Sample](https://public.tableau.com/static/images/Ex/ExampleDash/1_rss.png)

---

## Table of Contents

- About the project
- Key features
- Tech stack
- Data sources
- Repository layout
- Quick start
- Full pipeline: steps
  - Data cleaning
  - Feature engineering
  - Training (PyTorch)
  - Evaluation and explainability
  - Tableau dashboard
- Releases (download and run)
- Reproducibility and tips
- Tests and CI
- Contributing
- License
- Contact

---

## About the project

This repository demonstrates how to build a credit risk grading pipeline end to end. It uses a public dataset from Kaggle. The work covers:

- Clean raw loan records and prepare a robust dataset.
- Build engineered features that reflect borrower risk.
- Train a neural network classifier in PyTorch.
- Produce model metrics and calibration plots.
- Build an interactive Tableau dashboard that contrasts model scores with actual loan grades.

The project targets analysts and ML engineers working in credit risk or consumer lending.

---

## Key features

- Complete preprocessing pipeline for loan data.
- Feature engineering templates for categorical, numeric, temporal, and ratio features.
- PyTorch model code with modular training and inference.
- Evaluation scripts: confusion matrix, ROC, PR, calibration.
- Exportable prediction outputs for Tableau.
- Sample Tableau workbook and guide to reproduce the dashboard.
- Release bundle with packaged artifacts for quick demo.

---

## Tech stack

- Python 3.8+
- Pandas, NumPy, Scikit-learn
- PyTorch for modeling
- Matplotlib / Seaborn for plots
- Tableau Desktop / Tableau Public for dashboard
- Dockerfile for reproducible environment (optional)

Topics: classification, credit-risk-analysis, credit-scoring-and-classification, dashboards, data-cleaning, financial-analysis, loan-prediction, machine-learning, neural-networks, pytorch, tableau

---

## Data sources

Primary dataset: public Kaggle loan dataset (loan performance and borrower attributes). Use the original source from Kaggle for the raw CSV files. The repo contains processing code that assumes the common Kaggle structure: loans.csv, payments.csv, and lookup tables.

If you do not have the Kaggle files, run the notebook `notebooks/data_download_and_prep.ipynb` to load example snapshots that match the processing pipeline.

---

## Repository layout

- data/                - raw and processed sample data (large files not included)
- notebooks/           - exploratory notebooks and walkthroughs
- src/
  - data/              - cleaning and feature engineering modules
  - models/            - PyTorch model and training utilities
  - eval/              - evaluation and plotting scripts
  - export/            - code to export predictions for Tableau
- tableau/             - Tableau workbook (.twb / .twbx) and export guide
- Dockerfile           - optional container build
- requirements.txt
- LICENSE
- README.md

---

## Quick start

1. Clone the repo.
2. Create a Python environment and install dependencies.
3. Prepare data or use the included sample partitions.
4. Run the preprocessing script, train the model, and export predictions.

Example commands:

```bash
git clone https://github.com/aarmar22/credit-risk-scoring-end-to-end.git
cd credit-risk-scoring-end-to-end
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the full demo end to end (recommended on a machine with 8+ GB RAM):

```bash
python src/data/prepare_data.py --input data/raw --output data/processed
python src/models/train.py --config configs/train_config.yaml
python src/eval/generate_reports.py --preds outputs/predictions.csv --out reports/
```

Use the `notebooks/` folder to step through the work interactively.

---

## Full pipeline: steps

### Data cleaning

- Remove duplicates and impossible values.
- Impute missing values with domain-aware strategies.
- Convert date fields to consistent formats.
- Snap numeric distributions and cap outliers with winsorization.
- Persist processed dataset as Parquet for speed.

Sample snippet:

```python
from src.data.clean import clean_loans
cleaned = clean_loans("data/raw/loans.csv")
cleaned.to_parquet("data/processed/loans.parquet")
```

### Feature engineering

- Bucketize income, age, and loan amounts.
- Create DTI (debt-to-income) and utilization ratios.
- One-hot encode high-cardinality categorical fields with frequency thresholds.
- Build time-series aggregates where available (e.g., prior default counts).

Feature list example:
- dti_ratio
- log_loan_amount
- term_months
- employment_length_bucket
- credit_history_years

### Training (PyTorch)

The model uses a feed-forward network with embedding layers for categorical inputs. The code supports class weighting and focal loss.

Key files:
- src/models/model.py
- src/models/train.py
- configs/train_config.yaml

Run training:

```bash
python src/models/train.py --config configs/train_config.yaml
```

Model checkpoints and logs will appear in `outputs/`. The training script saves the final model as a .pt file.

### Evaluation and explainability

- Evaluate with ROC AUC, precision-recall, and confusion matrix.
- Create calibration plots to check score alignment with observed default rates.
- Use SHAP or permutation importance to show feature effects.

Example:

```bash
python src/eval/generate_reports.py --preds outputs/predictions.csv --labels data/processed/labels.csv --out reports/
```

Reports include:
- ROC and PR curves
- Confusion matrix at selected thresholds
- Calibration plot and Brier score
- Top feature importances

### Tableau dashboard

- Export model predictions and sample borrower metadata as CSV.
- Load the CSV into Tableau and use the workbook in `tableau/`.
- Dashboard views include score distribution, lift chart, cohort comparison, and case-level detail.

Export for Tableau:

```bash
python src/export/to_tableau.py --preds outputs/predictions.csv --meta data/processed/meta.csv --out tableau/tableau_export.csv
```

Open `tableau/credit_scoring_dashboard.twbx` in Tableau Desktop or Tableau Public to explore.

---

## Releases

Download the packaged release from this link:

https://github.com/aarmar22/credit-risk-scoring-end-to-end/releases

Visit that Releases page, download the appropriate release asset (for example `credit-risk-demo-v1.0.tar.gz` or `credit-risk-demo-v1.0.zip`), and run the included install or demo script. Example:

```bash
# after downloading the release asset
tar -xzf credit-risk-demo-v1.0.tar.gz
cd credit-risk-demo-v1.0
chmod +x run_demo.sh
./run_demo.sh
```

The release bundle contains:
- Preprocessed sample data
- Trained model checkpoint (.pt)
- A runnable demo script to generate predictions and the Tableau export
- A small README inside the bundle with exact file names

If the release link does not work, check the repository "Releases" section on GitHub.

---

## Reproducibility and tips

- Fix random seeds in configs: set seed for NumPy, Python, and PyTorch.
- Use the provided Dockerfile when you need exact environment parity.
- For large data, process in chunks and persist intermediate Parquet files.
- Monitor calibration across borrower segments (income, term, region).

Suggested improvements:
- Add target encoding with cross-validation to reduce leakage.
- Use monotonicity constraints in gradient boosting if business rules require them.
- Add adversarial validation to detect data shift between train and test.

---

## Tests and CI

- Basic unit tests live in `tests/`.
- Run tests with pytest:

```bash
pytest -q
```

- CI pipeline runs linting, unit tests, and a lightweight training smoke test.

---

## Contributing

- Open an issue for bugs or feature requests.
- Fork the repo and create a branch for your change.
- Keep changes small and focused. Add tests for data logic and model utilities.
- Follow PEP8 for Python code.

---

## License

This project uses the MIT License. See the LICENSE file for details.

---

## Contact

- GitHub: aarmar22 (owner)
- Issues: Use the repository Issues tab to report bugs or request features

https://github.com/aarmar22/credit-risk-scoring-end-to-end/releases