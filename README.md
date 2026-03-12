# KRA Tax Classification — Missing Tax Type Identification

A machine learning and rule-based system to identify and classify taxpayers across three tax categories: **VAT**, **Turnover Tax**, and **Personal Income Tax (PIT)**.

---

## 📌 Project Overview

The Kenya Revenue Authority (KRA) holds records of taxpayers who are missing tax type registrations. This project automates the identification of such taxpayers by analysing their transaction history and item data, then recommending the appropriate tax category for each.

Three analyses are carried out:

| Analysis | Approach | Objective |
|---|---|---|
| VAT Classification | Machine Learning | Classify items as VAT_TAXABLE or VAT_EXEMPT |
| Qualify for Turnover | Rule-based | Identify unregistered taxpayers eligible for Turnover Tax |
| Upgrade to PIT | Rule-based | Flag Turnover Tax holders who should upgrade to PIT |

---

## 🗂️ Project Structure

```
project/
│
├── QualifyForVAT.ipynb                  ← ML model: VAT item classification
├── QualifyForTurnover.ipynb             ← Rule-based: Turnover Tax eligibility
├── UpgradeFromTurnoverToPIT.ipynb       ← Rule-based: PIT upgrade identification
│
├── main.py                              ← Streamlit dashboard
│
├── missing_tax_type_taxpayer_sample_dataset_sales_and_purchases.csv
├── missing_tax_type_taxpayer_sample_dataset_imports_and_exports.csv
├── missing_tax_type_taxpayer_sample_dataset_registration.csv
│
├── tfidf_models_comparison.png          ← Model performance chart (generated)
├── tfidf_confusion_matrices.png         ← Confusion matrices (generated)
├── tfidf_model_comparison_results.xlsx  ← Model metrics export (generated)
└── KRA_tfidf_taxpayer_recommendations.xlsx  ← Taxpayer output (generated)
```

---

## 🔍 Analysis 1 — VAT Classification (Machine Learning)

### What it does
Reads item names from taxpayer transaction data and automatically classifies each item as **VAT_TAXABLE** or **VAT_EXEMPT** using text-based machine learning.

### How it works
1. Item names are converted to numerical features using **TF-IDF** and **spaCy** word vectors
2. Three classifiers are trained and compared: **SVM**, **Logistic Regression**, and **Random Forest**
3. The best model is applied to all 1,422 items in the dataset
4. Results are aggregated per taxpayer and exported to Excel

### Model Results

#### TF-IDF Features
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| SVM | 76.47% | 77.06% | 76.47% | 75.58% |
| Logistic Regression | 76.47% | 77.06% | 76.47% | 75.58% |
| **Random Forest** | **82.35%** | **82.44%** | **82.35%** | **82.09%** |

#### spaCy Features
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| SVM | 75.00% | 76.12% | 75.00% | 73.89% |
| Logistic Regression | 72.06% | 74.01% | 72.06% | 71.98% |
| **Random Forest** | **80.88%** | **81.34%** | **80.88%** | **80.51%** |

### ✅ Best Model: TF-IDF + Random Forest (F1 = 82.09%)

### Classification Output
| Class | Count | Share |
|---|---|---|
| VAT_TAXABLE | 1,392 | 97.9% |
| VAT_EXEMPT | 30 | 2.1% |
| **Total** | **1,422** | **100%** |

---

## 🔍 Analysis 2 — Qualify for Turnover Tax (Rule-based)

### What it does
Identifies unregistered taxpayers whose annual turnover falls within the Turnover Tax band, making them eligible for registration.

### Eligibility Rules
1. Annual sales fall within the Turnover Tax threshold band
2. Not already registered for Turnover Tax, PIT, or Corporate Income Tax (CIT)
3. Has recorded activity within the 12-month analysis window
4. Data sources considered: sales, purchases, imports, and exports

### Analysis Window
**03/2025 – 02/2026** (rolling 12 months)

### Result
> **215 unregistered taxpayers** identified as eligible for Turnover Tax registration.

---

## 🔍 Analysis 3 — Upgrade from Turnover Tax to PIT (Rule-based)

### What it does
Reviews taxpayers currently registered for Turnover Tax and flags those whose annual sales now exceed the upper Turnover Tax limit, making them liable for Personal Income Tax (PIT).

### Upgrade Rules
1. Currently registered with `turnover_tax == 1`
2. Annual sales (B2B + B2C) exceed the upper Turnover Tax threshold
3. Consistent sales data available by year

### Result
> Approximately **180 taxpayers** flagged for upgrade from Turnover Tax to PIT.

---

## 📊 Dashboard

An interactive Streamlit dashboard (`main.py`) visualises all three analyses in one place.

### Features
- Overview page with key metrics across all three analyses
- VAT model comparison — grouped bar charts, radar chart, confusion matrices
- Toggle between TF-IDF and spaCy results
- Turnover Tax eligibility funnel
- PIT upgrade pipeline and gender breakdown

### How to Run

**1. Install dependencies**
```bash
pip install streamlit plotly pandas numpy
```

**2. Run the dashboard**
```bash
python -m streamlit run main.py
```

**3. Open in browser**

Streamlit will automatically open `http://localhost:8501` in your browser.

---

## ⚙️ Requirements

```
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
spacy
streamlit
plotly
openpyxl
```

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn spacy streamlit plotly openpyxl
```

For spaCy language model:
```bash
python -m spacy download en_core_web_md
```

---

## 📁 Output Files

After running the notebooks, the following files are generated:

| File | Contents |
|---|---|
| `tfidf_model_comparison_results.xlsx` | Model metrics and test set predictions |
| `KRA_tfidf_taxpayer_recommendations.xlsx` | Taxpayer-level VAT recommendations |
| `tfidf_models_comparison.png` | Bar chart comparing all model metrics |
| `tfidf_confusion_matrices.png` | Confusion matrices for all 3 models |

---

## 🔑 Key Takeaways

- **Random Forest consistently outperforms** SVM and Logistic Regression on both TF-IDF and spaCy features
- **TF-IDF features outperform spaCy** on this dataset, likely due to the short, keyword-heavy nature of item names
- **97.9% of items are VAT_TAXABLE**, reflecting the nature of the taxpayer activity in the dataset
- **215 taxpayers** can be brought into the tax net through Turnover Tax registration
- Rule-based and ML approaches complement each other — ML handles unstructured text, rules handle structured thresholds

---

## 📝 Notes

- All taxpayer PINs in the dataset are masked for privacy
- Analysis window is based on the current date at time of running — results will update if notebooks are re-run in a later period
- The Streamlit dashboard uses hardcoded results from the last notebook run; re-run notebooks and update `main.py` values to refresh
