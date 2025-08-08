# Credit Risk Scoring & Dashboard

**Author:** Or Ben‑Haim  
**Stack:** Python · PyTorch · Scikit‑learn · Pandas · Tableau

## Overview
Predicting loan **risk grades** from applicant features and exploring results in a **Tableau dashboard**.  
Data is sourced from Kaggle and processed into a dataset that feeds the dashboard. 
The model is a **fully‑connected neural network (PyTorch)** for multi‑class classification, evaluated both technically and with business‑oriented views.

## Links
- **Kaggle dataset:** https://www.kaggle.com/datasets/laotse/credit-risk-dataset  
- **Public Tableau dashboard:** https://public.tableau.com/views/LoanGradePredictionAnalysis/LoanGradesPopulationInsightsModelvsActual

## Repository Structure
- `data/raw/` – Original Kaggle CSV  
- `data/processed/` – Processed dataset used by the dashboard (`loan_predictions_final_clean.xlsx`)  
- `dashboard/` – Tableau workbook (`Loan Grade Prediction Analysis.twb`) + screenshot  
- `notebooks/` – `End to End Credit Risk Scoring.ipynb` (cleaning → modeling → export)  
- `reports/` – Final PDF report

## Model
- **Type:** Fully‑connected neural network (feed‑forward MLP) in **PyTorch**  
- **Task:** Multi‑class loan grade classification  
- **Label handling:** Collapsed rare grades to 4 classes (A–D) for stability  
- **Highlights:** ~88% accuracy after collapsing; most confusion between adjacent grades (A↔B, C↔D)

## How to Reproduce
1. Open `notebooks/End to End Credit Risk Scoring.ipynb` and run the steps to produce `data/processed/loan_predictions_final_clean.xlsx`.  
2. Open `dashboard/Loan Grade Prediction Analysis.twb` in Tableau Desktop (or use the public link above) to explore **Model vs Actual** views.

## Results (short)
- Balanced performance across classes after label merge (~88% overall accuracy).  
- Dashboard surfaces error patterns and the impact on loan amounts and interest rates.

## Data Source
Kaggle — **Credit Risk Dataset** by laotse: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

## License
MIT © 2025 
