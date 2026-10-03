# Google Search Ranking & Discoverability Capstone
**FlyRank Applied Search Intelligence Machine Learning Track — Week 8 Capstone Deliverable**

[![Paper Live](https://img.shields.io/badge/Deployed_Paper-Live_Page-blue.svg)](https://ilyan321.github.io/google-search-ranking-capstone/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Credit](https://img.shields.io/badge/Data_Credit-FlyRank_AI-emerald.svg)](https://flyrank.ai)

---

## 📄 Final Research Paper
- **Direct Live Paper URL:** [https://ilyan321.github.io/google-search-ranking-capstone/](https://ilyan321.github.io/google-search-ranking-capstone/)
- **Submission Pointer File:** `submission/paper_url.txt`

---

## 🎯 Executive Summary & Core Results

This capstone builds a machine learning-driven **Content Refresh Opportunity Scoring Engine** on 30,000 anonymized enterprise content assets across 32 clients from the FlyRank Search Intelligence Warehouse release.

### Verified Empirical Benchmarks
| Validation Metric | Rule Heuristic Baseline | ElasticNet Logistic Reg | Random Forest (Ensemble) | HistGradientBoosting (LightGBM) |
|---|---|---|---|---|
| **ROC-AUC** | 0.7416 | 0.8856 | **0.9993** | **1.0000** |
| **PR-AUC** | 0.6011 | 0.8002 | **0.9988** | **0.9999** |
| **Precision@10** | 0.7000 | 0.9000 | **1.0000** | **1.0000** |
| **Precision@20** | 0.7000 | 0.9500 | **1.0000** | **1.0000** |
| **Precision@50** | 0.7600 | 0.9800 | **1.0000 (+31.5%)** | **1.0000 (+31.5%)** |
| **Brier Score** | 0.2236 | 0.1346 | **0.0444** | **0.0023** |

*All metrics validated under 5-Fold GroupKFold cross-validation partitioned across client domains (zero website leakage).*

---

## 🔍 Key Signal Analysis & Myth-Busters
1. **Third-Party Keyword Volume vs. GSC Impressions ($r = 0.0012$):** Pearson correlation between keyword search volume and actual 90-day impressions is near zero. Keyword tools fail to capture multi-query long-tail fanout and SERP feature suppression.
2. **Word Count is Non-Differentiating:** Growing pages have a median word count of **~2,848 words** vs. declining pages at **~2,909 words**. Arbitrary word count inflation does not reverse ranking decay.
3. **Empirical CTR Degradation:** Click-Through Rate drops from **1.48%** (Top 3) to **0.32%** (Positions 8–20 striking distance) and **0.15%** (deep positions). Striking distance pages with sub-benchmark CTR represent high-ROI metadata optimization opportunities.

---

## 📂 Repository Structure

```
.
├── submission/
│   └── paper_url.txt                         # Mandatory direct live deployed paper URL
├── work/
│   ├── 01_signal_analysis_foundations.ipynb   # Week 1-2 Signal Analysis & Myth-Busting
│   ├── 02_content_archetype_clustering.ipynb # Week 3-4 K-Means Archetype Clustering & PCA
│   ├── 03_warehouse_duckdb_feature_engineering.ipynb # Week 5-6 DuckDB SQL Aggregations & Anti-Leakage
│   └── 04_capstone_opportunity_scoring.ipynb # Week 7-8 Capstone ML Pipeline & Reason Engine
├── docs/
│   ├── index.html                            # Publication-grade interactive research paper
│   └── figures/                              # Rendered high-resolution charts & JSON exports
├── figures/                                  # Pipeline figures & JSON metrics
├── scripts/
│   ├── fetch_data.py                         # Hugging Face dataset download script
│   ├── train_and_evaluate.py                 # Full ML training & out-of-fold validation pipeline
│   └── generate_notebooks.py                 # Automated assignment notebook generator
├── index.html                                # Root copy of the research paper
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## 🚀 Quickstart & Reproducibility

```bash
# 1. Clone repository
git clone https://github.com/Ilyan321/google-search-ranking-capstone.git
cd google-search-ranking-capstone

# 2. Setup Python environment with uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Run full training and evaluation pipeline
python scripts/train_and_evaluate.py

# 4. Explore Jupyter notebooks
jupyter lab work/
```

---

## 🔒 Public Safety & Privacy Compliance
This research adheres strictly to the FlyRank Internship Data Use Terms:
- No client brand names, domain names, or unmasked URLs.
- Entity IDs (`content_id`, `client_id`) are salted cryptographic hashes.
- No client-identifying exports or raw queries.
- Honest, non-causal decision-support framing (no claims of reverse-engineering Google algorithms).

---

## 🤝 Acknowledgments & Data Credit
Built on the **[FlyRank ML Internship dataset](https://flyrank.ai)**. Special thanks to the FlyRank AI team for curating the enterprise search intelligence warehouse.
